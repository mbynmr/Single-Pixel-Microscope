import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.ndimage import rotate
import cv2
import pyvisa as visa
from tqdm import tqdm


class Camera:
    def __init__(self, resolution):
        self.resolution = resolution
        pixels = np.asarray(self.resolution).prod()  # 32*32 = 1024
        DMD_resolution = [608, 684]
        self.factor = int(DMD_resolution[0] / (self.resolution[0] * 2 ** (1 / 2)))
        # todo factor: floor, and pad vertically to stop rounding
        self.pad_width = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor, self.factor * 2))).shape[0])

        self.factor_old = int(DMD_resolution[0] / self.resolution[0])  # todo remove?
        self.pad_width_old = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor_old, self.factor_old * 2))).shape[0])  # todo remove?
        # pad_width_old = 760

        # Hadamard matrix
        self.hadamard_mat = hadamard(pixels)
        self.hadamard_all = np.zeros([pixels * 2, pixels])
        self.hadamard_all[0::2, ...] = (1 + self.hadamard_mat) / 2  # 1 & 0
        self.hadamard_all[1::2, ...] = (1 - self.hadamard_mat) / 2  # 0 & 1
        # self.hadamard_all = 1 - self.hadamard_all  # todo check if 1 is toward the sample and 0 is away or not

        # set up fullscreen cv2 output (for masks on the DMD)
        # self.display_time = 3 * (1 / 60)  # in seconds = 3 * (1 / 60)
        # self.display_time_ms = int(self.display_time * 1e3)  # in milliseconds
        # self.display_time_ms = int(1000 / 60)  # in milliseconds
        self.window = set_up_mask_output()

        # set up the multimeter & measurements
        plc = 1 / 50  # Power Line Cycle: the time period of one oscillation of the power line at 50Hz
        xplc = [0.02, 0.1, 1, 10]  # these are the only options for the multimeter
        self.xplc = xplc[2]  # measure every ___ power line cycle
        self.rm, self.multimeter = set_up_multimeter()
        # prepare the multimeter for a run of measurements
        self.reset_multimeter()
        # set the measurement time
        # self.measurements_time = (self.display_time_ms * 3) / 1000  # in seconds, approximately 3 frames long
        self.minimum_measurements_per_mask = 3  # todo how many measurements per mask?
        self.integration_time = plc * self.xplc  # in seconds, the time for a single measurement
        self.measurements_time = self.integration_time * self.minimum_measurements_per_mask  # in seconds

    def take_picture(self):
        measurements_p_and_m = self.measure()
        # measurements_p_and_m = np.loadtxt("outputs/measurements.txt")
        measurements = measurements_p_and_m[0::2, ...] - measurements_p_and_m[1::2, ...]  # differential measurement

        # reconstruct image from measurements and masks
        image = self.reconstruct(measurements)
        plt.imsave(f"outputs/first_output.png", image, cmap=plt.get_cmap('gray'))

    def reconstruct(self, measurements):
        image = self.hadamard_mat @ measurements
        image = image - np.amin(image)
        image = image / np.amax(image)
        return np.uint8(image.reshape(self.resolution) * 255)  # todo formatting

    def measure(self):
        # the first cv2 display takes a while to set up, and this was causing problems with the measurements
        cv2.imshow(self.window, cv2.imread("originals/splash.png"))  # first cv2 display
        cv2.waitKey(1)
        time.sleep(15)

        measurements = np.zeros([self.hadamard_all.shape[0]])
        deviations = np.zeros([self.hadamard_all.shape[0]])
        numbers = np.zeros([self.hadamard_all.shape[0]])
        # times = np.zeros([self.hadamard_all.shape[0]])
        for i, mask in enumerate(self.hadamard_all[:, ...]):  # todo first 20 only!  # todo are we taking the wrong dimension?
            # convert the plus & minus hadamard matrixes into the correct images to be displayed on the DMD
            mask_show = self.reshape_mask(mask)

            # convert to uint8 and show on the window  # todo uint8 or binary?
            cv2.imshow(self.window, np.uint8(mask_show * 255))
            # show the window (for the display time)
            cv2.waitKey(1)
            # time.sleep(60 / 1000)  # ~17ms, to make sure this mask is displaying before we take any measurements

            # time.sleep(self.display_time_ms)  # todo testing
            time.sleep(self.integration_time * 3)  # todo testing
            # tell the multimeter to wait for a trigger (which happens instantly since the trigger is set to immediate)
            self.multimeter.write('INITiate')
            # string = self.multimeter.query('FETCh?')
            # fetch the data from the instruments internal storage, format them, then take the average of them
            # measurements[i] = np.mean(np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float))

            time.sleep(self.measurements_time)  # wait for a certain amount of time to take a multiple measurements
            buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
            deviations[i] = np.std(buffer)
            measurements[i] = np.mean(buffer)
            while deviations[i] > 1e-5 * measurements[i] or len(buffer) < self.minimum_measurements_per_mask:
                time.sleep(self.integration_time)
                buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
                deviations[i] = np.std(buffer)
                measurements[i] = np.mean(buffer)
                # print(f"\n{i = } and buffer length = {len(buffer)}")
                # print(string)
                # print(buffer)
            # times[i] = time.time() - start
            numbers[i] = len(buffer)  # int(np.shape(buffer)[0])
            # 1*10^-3 or 4*10^-4
            # 4*10^-7 ish

        np.savetxt("outputs/numbers.txt", numbers, fmt='%d')
        np.savetxt("outputs/measurements.txt", measurements)
        np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/times.txt", times[:200])
        # np.savetxt("outputs/measurements_p_and_m.txt", measurements[:200])
        return measurements

    def reshape_mask(self, mask):
        # reshape to be 2D and square, rotate 45, integer upscale, pad to be rectangular, rotate 90, invert (1 - mask)
        return 1 - np.rot90(np.pad(np.kron(
            rotate(mask.reshape(self.resolution), angle=45), np.ones((self.factor, self.factor * 2))
        ), ((0, 0), (self.pad_width * 3 / 2, self.pad_width / 2))), axes=(1, 0))  # 1976x608

    def reshape_mask_old(self, mask):  # no 45 degree rotation to eliminate grid artefact, also readable code!
        # turn 32x32 masks into 608x1216 by integer scaling
        mask_show = np.kron(mask.reshape(self.resolution), np.ones((self.factor_old, self.factor_old * 2)))
        # pad the rectangular mask with zeros on both sides, so the on the DMD it appears as a centred square
        mask_show = np.pad(mask_show, ((0, 0), (self.pad_width_old - 380, 380)))  # 608x1976
        # the DMD rotates the image 90 degrees clockwise, so we do the opposite
        # also, the DMD points 0 toward our sample and 1 away, so we need to invert our masks
        mask_show = 1 - np.rot90(mask_show, axes=(1, 0))  # 1976x608
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
