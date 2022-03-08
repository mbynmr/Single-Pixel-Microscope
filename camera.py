import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import cv2
import pyvisa as visa
from tqdm import tqdm


class Camera:
    def __init__(self, resolution):
        self.resolution = resolution
        pixels = np.asarray(self.resolution).prod()  # 32*32 = 1024
        self.DMD_resolution = [608, 684]
        self.factor = int(self.DMD_resolution[0] / self.resolution[0])  # todo floor, and pad vertically to stop rounding
        self.pad_width = int((self.DMD_resolution[1] * 2 -
                              np.kron(np.zeros(self.resolution), np.ones((self.factor, self.factor * 2))).shape[0]))
        # pad_width = 760

        # Hadamard matrix
        self.hadamard_mat = hadamard(pixels)
        self.hadamard_all = np.zeros([pixels * 2, pixels])
        self.hadamard_all[0::2, ...] = (1 + self.hadamard_mat) / 2  # 1 & 0
        self.hadamard_all[1::2, ...] = (1 - self.hadamard_mat) / 2  # 0 & 1
        # self.hadamard_all = 1 - self.hadamard_all  # todo check if 1 is toward the sample and 0 is away or not

        # set up fullscreen cv2 output (masks on the DMD)
        self.display_time = 3 * (1 / 60)  # in seconds = 3 * (1 / 60)
        self.display_time_ms = int(self.display_time * 1e3)  # in milliseconds
        # self.display_time_ms = int(1000 / 60)  # in milliseconds
        self.window = set_up_mask_output()

        # set up the multimeter
        # self.rm, self.multimeter = set_up_multimeter()
        # prepare the multimeter for a run of measurements
        # self.reset_multimeter()
        # set the measurement time
        # self.measurement_time = (self.display_time_ms * 3) / 1000  # in seconds, approximately 3 frames long
        number_of_measurements_per_mask = 2  # todo 10?
        self.measurement_time = (1 / 50) * 0.02 * number_of_measurements_per_mask  # in seconds

    def take_picture(self):
        # measurements_p_and_m, times = self.measure()
        # np.savetxt("outputs/measurements_p_and_m.txt", measurements_p_and_m)
        # np.savetxt("outputs/times.txt", times)
        times = np.loadtxt("outputs/times.txt")
        measurements_p_and_m = np.loadtxt("outputs/measurements_p_and_m.txt")
        plt.plot(times, measurements_p_and_m)
        plt.show()
        # measurements = measurements_p_and_m[0::2, ...] - measurements_p_and_m[1::2, ...]  # differential measurement

        # reconstruct image from measurements and masks
        # image = self.reconstruct(measurements)
        # plt.imsave(f"outputs/first_output.png", image, cmap=plt.get_cmap('gray'))

    def reconstruct(self, measurements):
        image = self.hadamard_mat @ measurements
        return np.uint8(image.reshape(self.resolution) * 255)  # todo formatting

    def measure(self):
        # the first cv2 display takes a while to set up, and this was causing problems with the measurements
        cv2.imshow(self.window, cv2.imread("originals/splash.png"))  # first cv2 display
        cv2.waitKey(self.display_time_ms)
        time.sleep(5)
        start = time.time()

        measurements = np.zeros([self.hadamard_all.shape[0]])
        deviations = np.zeros([self.hadamard_all.shape[0]])
        numbers = np.zeros([self.hadamard_all.shape[0]])
        times = np.zeros([self.hadamard_all.shape[0]])
        for i, mask in enumerate(self.hadamard_all[:, ...]):  # todo first 20 only!
            # convert the plus & minus hadamard matrixes into the correct images to be displayed on the DMD
            # turn 32x32 masks into 608x1216 by integer scaling
            mask_show = np.kron(mask.reshape(self.resolution), np.ones((self.factor, self.factor * 2)))
            # pad the rectangular mask with zeros on both sides, so the on the DMD it appears as a centred square
            mask_show = np.pad(mask_show, ((0, 0), (self.pad_width - 400, 400)))  # 608x1976  # todo 400 ish
            # the DMD rotates the image 90 degrees clockwise, so we need to do the opposite
            mask_show = np.rot90(mask_show, axes=(1, 0))  # 1976x608

            # convert to uint8 and show on the window
            cv2.imshow(self.window, 0 * np.uint8(mask_show * 255))
            # show the window (for the display time)
            cv2.waitKey(self.display_time_ms)
            # time.sleep(17 * 1e-3)  # 17ms, to make sure this mask is displaying before we take any measurements

            # tell the multimeter to wait for a trigger (which happens instantly since the trigger is set to immediate)
            self.multimeter.write('INITiate')
            time.sleep(self.measurement_time)  # wait for a certain amount of time to take a multiple measurements

            # fetch the data from the instruments internal storage, format them, then take the average of them
            data = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
            times[i] = time.time() - start
            numbers[i] = int(np.shape(data)[0])
            deviations[i] = np.std(data)
            measurements[i] = np.mean(data)

        np.savetxt("outputs/numbers.txt", numbers)
        np.savetxt("outputs/measurements.txt", measurements)
        np.savetxt("outputs/deviations.txt", deviations)
        return measurements, times

    def reset_multimeter(self):
        # set the multimeter up to take (and store) infinite measurements as fast as possible when called to INITiate

        # tell the multimeter to clear the buffer and reset
        self.multimeter.write('*RST')

        # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
        self.multimeter.write('CONFigure:VOLTage:DC 10, MIN')  # todo 10V is too high, right?

        # set the number of PLC for each integration - i.e. measurement time
        self.multimeter.write('SENSe:VOLTage:DC:NPLCycles MINimum')
        # MINimum PLC = 0.02, so a measurement happens every 0.02 * (1 / 50) = 0.0004 seconds, a rate of 2.5kHz

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
    multimeter.timeout = None  # sets no instrument timeout
    return rm, multimeter
