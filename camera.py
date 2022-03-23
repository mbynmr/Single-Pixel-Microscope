import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.ndimage import rotate
import cv2
import pyvisa as visa
from HadamardOrdering import Walsh
from tqdm import tqdm


class Camera:
    def __init__(self, resolution, xplc, measurements, fraction, ordering):
        self.resolution = resolution
        self.xplc = xplc
        self.minimum_measurements_per_mask = measurements
        self.frac = fraction
        self.ordering = ordering

        pixels = np.asarray(self.resolution).prod()  # 32*32 = 1024
        DMD_resolution = [608, 684]

        self.factor = int(DMD_resolution[0] / (self.resolution[0] * 2 ** (1 / 2)))
        # todo factor: floor, and pad vertically to stop rounding
        self.pad_width = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor, self.factor * 2))).shape[0])
        # todo remove _old?
        self.factor_old = int(DMD_resolution[0] / self.resolution[0])
        self.pad_width_old = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor_old, self.factor_old * 2))).shape[0])
        # pad_width_old = 760

        if self.ordering == 'Hadamard':  # Hadamard matrix
            self.matrix = hadamard(pixels)
            # self.hadamard_all = 1 - self.hadamard_all  # todo check if 1 is toward the sample and 0 is away or not
        elif self.ordering == 'Walsh':  # Walsh ordering Hadamard matrix
            self.matrix = Walsh(self.resolution[0], hadamard(pixels))
        elif self.ordering == 'Random':
            self.matrix = (np.random.randint(low=0, high=2, size=[pixels, pixels]) - 0.5) * 2
            # for i, column in enumerate(self.matrix):  # todo make sure 50/50 1/-1
            #     self.matrix[i, ...] = column
        else:
            raise NotImplementedError(f"there is no '{ordering}' matrix")

        self.matrix_all = np.zeros([pixels * 2, pixels])
        self.matrix_all[0::2, ...] = (1 + self.matrix) / 2  # 1 & 0
        self.matrix_all[1::2, ...] = (1 - self.matrix) / 2  # 0 & 1

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

    def take_picture(self, pause_time):
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
        return np.uint8(image.reshape(self.resolution)[..., ::-1] * 255)  # todo ::-1
        # return np.uint8(image.reshape(self.resolution) * 255)

    def measure(self, pause_time):
        # the first cv2 display takes a while to set up, and this was causing problems with the measurements
        cv2.imshow(self.window, cv2.imread("originals/splash.png"))  # first cv2 display
        cv2.waitKey(17)
        time.sleep(pause_time)

        measurements = np.zeros([self.matrix_all.shape[0]])
        deviations = np.zeros([self.matrix_all.shape[0]])
        numbers = np.zeros([self.matrix_all.shape[0]])
        # times = np.zeros([self.hadamard_all.shape[0]])
        # for i, mask in enumerate(self.hadamard_all[:, ...]):

        i = 0
        while i < 2 * int(self.matrix_all.shape[0] * self.frac / 2):
            # mask_show = self.reshape_mask_old(self.matrix_all[i, ...])
            mask_show = self.reshape_mask(self.matrix_all[i, ...])
            cv2.imshow(self.window, np.uint8(mask_show * 255))
            # show the window (for the display time)
            cv2.waitKey(17 * 3)
            # time.sleep(17 * 1e-3)  # 17ms > (1/60)s, to make sure this mask is displaying before taking measurements
            time.sleep(self.integration_time * 3)

            # tell the multimeter to wait for a trigger (which happens instantly since the trigger is set to immediate)
            self.multimeter.write('INITiate')
            time.sleep(self.measurements_time)  # wait for a certain amount of time to take a multiple measurements

            try:
                # fetch the data from the instruments internal storage and format them
                buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
                deviations[i] = np.std(buffer)
                measurements[i] = np.mean(buffer)
                while deviations[i] > 0.05 * measurements[i] or len(buffer) < self.minimum_measurements_per_mask:
                    # if deviations[i] > 0.05 * measurements[i]:
                    if np.shape(buffer)[0] > 10 * self.minimum_measurements_per_mask:
                        i -= 1
                        print(f"redoing mask {i}")
                        break
                        # print(f"{i}, deviation too big, {np.shape(buffer)}")
                    time.sleep(self.integration_time)
                    # string = self.multimeter.query('FETCh?')
                    buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
                    deviations[i] = np.std(buffer)
                    measurements[i] = np.mean(buffer)
                # times[i] = time.time() - start
                numbers[i] = len(buffer)  # int(np.shape(buffer)[0])
            except UnicodeDecodeError:
                print("u got an error, let's ignore that and move on thanks")
                i -= 1
            i += 1

        np.savetxt("outputs/numbers.txt", numbers, fmt='%d')
        np.savetxt("outputs/measurements.txt", measurements)
        np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/times.txt", times[:200])
        return measurements

    def reshape_mask(self, mask):
        # reshape to be 2D and square, rotate 45, integer upscale, pad to be rectangular, rotate 90, invert (1 - mask)
        # return 1 - np.rot90(np.pad(np.kron(
        #     mask.reshape(self.resolution), np.ones((self.factor, self.factor * 2))
        # ), ((0, 0), (int(self.pad_width / 2), int(self.pad_width / 2)))), axes=(0, 1))  # 1976x608
        mask = np.asarray(rotate(mask.reshape(self.resolution), angle=45), dtype=np.int_)
        return 1 - np.rot90(np.pad(np.kron(
            mask, np.ones((self.factor, self.factor * 2))
        ), ((0, 0), (int(self.pad_width / 2), int(self.pad_width / 2)))), axes=(0, 1))  # 1976x608

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
        self.multimeter.write('DISPlay OFF')  # todo un-comment

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

        self.multimeter.write('DISPlay ON')  # todo do we need this

        # clear and close the instance of the instrument, and close the resource manager
        self.multimeter.write('*CLS')  # tell the multimeter to reset
        time.sleep(1)
        self.multimeter.clear()
        self.multimeter.close()
        self.rm.close()


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
