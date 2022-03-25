import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import cv2
import pyvisa as visa
from tqdm import tqdm

from HadamardOrdering import Walsh


class Camera:
    def __init__(self, resolution, xplc, measurements, fraction, ordering):
        self.resolution = resolution
        self.xplc = xplc
        self.minimum_measurements_per_mask = measurements
        self.frac = fraction
        self.ordering = ordering

        pixels = np.asarray(self.resolution).prod()  # 32*32 = 1024
        DMD_resolution = [684, 608]
        HDMI_resolution = DMD_resolution[::-1]

        self.factor = int((HDMI_resolution[1] + 1) / (self.resolution[0] * 2))
        self.pad = [int((HDMI_resolution[0] - (self.factor * self.resolution[1])) / 2),
                    int((HDMI_resolution[1] - (self.factor * self.resolution[1] * 2 - 1)) / 2)]
        # todo odd padding makes us 1 pixel off
        # self.factor = int(DMD_resolution[0] / self.resolution[0])
        # todo factor: floor, and pad vertically to stop rounding
        self.pad_width = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor, self.factor * 2))).shape[0])
        # todo remove _old?
        self.factor_old = int(DMD_resolution[0] / self.resolution[0])
        self.pad_width_old = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor_old, self.factor_old * 2))).shape[0])
        # pad_width_old = 760

        if self.ordering == 'Hadamard_Natural':  # Hadamard matrix
            self.matrix = hadamard(pixels)
            # self.hadamard_all = 1 - self.hadamard_all  # todo check if 1 is toward the sample and 0 is away or not
        elif self.ordering == 'Hadamard_Walsh':  # Walsh ordering Hadamard matrix
            self.matrix = Walsh(self.resolution[0], hadamard(pixels))
        elif self.ordering == 'Random':
            self.matrix = (np.random.randint(low=0, high=2, size=[pixels, pixels]) - 0.5) * 2
            # for i, column in enumerate(self.matrix):  # todo make sure 50/50 1/-1
            #     self.matrix[i, ...] = column
        else:
            raise NotImplementedError(f"there is no '{self.ordering}' matrix")

        matrix_both = np.zeros([pixels * 2, pixels])
        matrix_both[0::2, ...] = (1 + self.matrix) / 2  # 1 & 0
        matrix_both[1::2, ...] = (1 - self.matrix) / 2  # 0 & 1

        # self.matrix_all = np.zeros([pixels * 2, self.resolution[0], self.resolution[1]])
        # # 2x number of masks, each with length (n x n)
        # for i, mask in tqdm(enumerate(matrix_both)):
        #     self.matrix_all[i, ...] = (mask.reshape(self.resolution))
        self.matrix_all = matrix_both.reshape([2 * pixels, *self.resolution])  # todo check this reshapes how we think!

        # set up fullscreen cv2 output (masks on the DMD)
        self.window = set_up_mask_output()

        # set up the multimeter
        plc = 1 / 50
        self.rm, self.multimeter = set_up_multimeter()
        # prepare the multimeter for a run of measurements
        self.reset_multimeter()
        # set the measurement time
        # self.measurements_time = (self.display_time_ms * 3) / 1000  # in seconds, approximately 3 frames long
        self.integration_time = plc * self.xplc  # in seconds
        self.measurements_time = self.integration_time * self.minimum_measurements_per_mask  # in seconds

    def take_picture(self, pause_time=15):
        measurements_p_and_m = self.measure(pause_time)
        # measurements_p_and_m = np.loadtxt("outputs/measurements.txt")
        measurements = measurements_p_and_m[0::2, ...] - measurements_p_and_m[1::2, ...]  # differential measurement

        # reconstruct image from measurements and masks
        image = self.reconstruct(measurements)
        plt.imsave(f"outputs/SPC_image_{self.resolution[0]}_{self.xplc}"
                   f"_{self.minimum_measurements_per_mask}_{self.frac}_{self.ordering}.png",
                   image, cmap=plt.get_cmap('gray'))  # save image
        plt.imsave(f"outputs/SPC_image_{self.resolution[0]}_{self.xplc}"
                   f"_{self.minimum_measurements_per_mask}_{self.frac}_{self.ordering}_upscaled.png",
                   np.kron(image, np.ones((10, 10))), cmap=plt.get_cmap('gray'))  # 10x integer upscaled image

    def reconstruct(self, measurements):
        image = self.matrix @ measurements
        image = image - np.amin(image)
        image = image / np.amax(image)
        return np.uint8(image.reshape(self.resolution)[..., ::-1] * 255)

    def measure(self, pause_time):
        # the first cv2 display takes a while to set up, and this was causing problems with the measurements
        cv2.imshow(self.window, cv2.imread("originals/splash.png"))  # first cv2 display
        cv2.waitKey(17)
        time.sleep(pause_time)

        measurements = np.zeros([self.matrix_all.shape[0]])
        deviations = np.zeros([self.matrix_all.shape[0]])
        numbers = np.zeros([self.matrix_all.shape[0]])
        # times = np.zeros([self.matrix_all.shape[0]])

        next_mask = np.uint8(self.reshape_mask(self.matrix_all[0, ...]) * 255)
        i = 0
        while i < 2 * int(self.matrix_all.shape[0] * self.frac / 2):
            cv2.imshow(self.window, next_mask)
            # show the window (for the display time)
            cv2.waitKey(17)
            time.sleep(1 * (1 / 60))  # (1/60)s, to make sure this mask is displaying before we take any measurements
            # todo 2 * (1/ 60)? How low can we push this

            # tell the multimeter to wait for a trigger (which happens instantly since the trigger is set to immediate)
            self.multimeter.write('INITiate')
            start = time.time()
            if i < self.matrix_all.shape[0] - 1:
                next_mask = np.uint8(self.reshape_mask(self.matrix_all[i + 1, ...]) * 255)
            sleep_time = self.measurements_time - (time.time() - start)
            if sleep_time > 0:  # wait for the remaining amount of time to take a multiple measurements
                print(f"slept for {sleep_time}s on mask {i}")
                time.sleep(sleep_time)
            else:
                print(f"did not sleep on mask {i}")  # todo remove one or both of these prints
            try:
                # fetch the data from the instruments internal storage and format them
                buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
                deviations[i] = np.std(buffer)
                measurements[i] = np.mean(buffer)
                numbers[i] = len(buffer)  # int(np.shape(buffer)[0])
                while deviations[i] > 0.05 * measurements[i] or len(buffer) < self.minimum_measurements_per_mask:
                    # if deviations[i] > 0.05 * measurements[i]:
                    if np.shape(buffer)[0] > 10 * self.minimum_measurements_per_mask:
                        print(f"redoing mask {i}")
                        i -= 1
                        break
                    time.sleep(self.integration_time)
                    # string = self.multimeter.query('FETCh?')
                    buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
                    deviations[i] = np.std(buffer)
                    measurements[i] = np.mean(buffer)
                    numbers[i] = len(buffer)  # int(np.shape(buffer)[0])
                    # times[i] = time.time() - start
            except UnicodeDecodeError:
                print(f"got a UnicodeDecodeError on mask {i}, let's ignore that and repeat this mask")
                i -= 1
            i += 1

        np.savetxt("outputs/numbers.txt", numbers, fmt='%d')
        np.savetxt("outputs/measurements.txt", measurements)
        np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/times.txt", times[:200])
        return measurements

    def reshape_mask(self, mask):
        # reshape by: integer upscale, diamond grid correction, rotate 90, pad to be the correct HDMI shape
        return np.pad(np.rot90(my_45(np.kron(mask, np.ones((self.factor, self.factor)))), axes=(0, 1)),
                      ((self.pad[0], self.pad[0]), (self.pad[1], self.pad[1] + 1)))
        # todo is the 90 degree rotation necessary? was it upside down? this could be corrected on the final image?

    def reshape_mask_old(self, mask):  # no 45 degree rotation to eliminate grid artefact, also readable code!
        # turn 32x32 masks into 608x1216 by integer scaling
        mask_show = np.kron(mask.reshape(self.resolution), np.ones((self.factor_old, self.factor_old * 2)))
        # pad the rectangular mask with zeros on both sides, so the on the DMD it appears as a centred square
        mask_show = np.pad(mask_show, ((0, 0), (self.pad_width_old - 380, 380)))  # 608x1976
        # the DMD rotates the image 90 degrees clockwise, so we do the opposite
        # also, the DMD points 0 toward our sample and 1 away, so we need to invert our masks
        mask_show = 1 - np.rot90(mask_show, axes=(0, 1))  # 1976x608
        return mask_show

    def reset_multimeter(self):
        # set the multimeter up to take (and store) infinite measurements as fast as possible when called to INITiate

        # tell the multimeter to clear the buffer and reset
        self.multimeter.write('*RST')

        # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
        self.multimeter.write('CONFigure:VOLTage:DC 0.1, MIN')  # todo 0.1V = 100mV

        # set the number of PLC for each integration - i.e. measurement time
        # self.multimeter.write('SENSe:VOLTage:DC:NPLCycles MINimum')  # todo
        self.multimeter.write(f'SENSe:VOLTage:DC:NPLCycles {self.xplc}')
        # MINimum PLC = 0.02, so a measurement happens every 0.02 * (1 / 50) = 0.0004 seconds, a rate of 2.5kHz

        # turn the screen off so we don't get light from it affecting the measurements
        self.multimeter.write('DISPlay OFF')

        # set the trigger source to internal, and immediate, and set the trigger delay to auto
        self.multimeter.write('TRIGger:SOURce IMMediate')
        self.multimeter.write('TRIGger:DELay:AUTO ON')
        # set the instrument to take 1 sample after the trigger event
        self.multimeter.write('SAMPle:COUNt 1')
        # infinite triggers
        self.multimeter.write('TRIG:COUN INF')

    def close(self):
        # close the cv2 fullscreen window
        cv2.destroyAllWindows()

        self.multimeter.write('DISPlay ON')

        # clear and close the instance of the instrument, and close the resource manager
        self.multimeter.write('*CLS')  # tell the multimeter to reset
        time.sleep(1)
        self.multimeter.clear()
        self.multimeter.close()
        self.rm.close()


def my_45(mask):
    n = mask.shape[0]  # mask is n x n
    m = n - 1  # ((2 * n - 1) - 1) / 2 = m, which is the number of diagonals above (or below) the middle
    my_mask = np.zeros([2 * n - 1, n])  # my_mask is (2n - 1) x n

    my_mask[m, :] = np.diag(mask, k=0)  # middle row

    for d in range(m):  # first rows
        my_mask[d, int(n / 2 - d / 2):int(n / 2 - d / 2 + d + 1)] = np.diag(mask, k=(m - d))

    for d in range(m):  # last rows
        my_mask[2 * m - d, int(n / 2 - d / 2):int(n / 2 - d / 2 + d + 1)] = np.diag(mask, k=(d - m))

    return my_mask


def set_up_mask_output():
    window = 'output_window'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return window


def set_up_multimeter():
    # open an instance of the resource manager class and assign the handle rm
    rm = visa.ResourceManager()
    # open an instance of the USB resource class and assign the object handle multimeter
    multimeter = rm.open_resource('USB0::0x05E6::0x2100::1269989::INSTR')
    # USB0::0x05E6::0x2100::1269989::INSTR
    multimeter.timeout = None  # sets no instrument timeout
    return rm, multimeter
