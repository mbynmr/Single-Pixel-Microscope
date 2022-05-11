import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.interpolate import griddata, interp2d  # todo
import cv2
import pyvisa as visa
from tqdm import tqdm

from HadamardOrdering import Walsh, random_masks


class Camera:
    def __init__(self, resolution, xplc, measurements, fraction, method):
        self.resolution = resolution
        self.xplc = xplc
        self.minimum_measurements_per_mask = measurements
        self.frac = fraction
        self.method = method
        self.seed = np.random.randint(999)

        pixels = np.asarray(self.resolution).prod()  # 32*32 = 1024
        DMD_resolution = [684, 608]
        HDMI_resolution = DMD_resolution[::-1]
        self.array = np.zeros(HDMI_resolution)

        self.factor = int((HDMI_resolution[1] + 1) / (self.resolution[0] * 2))
        self.pad = [int((HDMI_resolution[0] - (self.factor * self.resolution[1])) / 2),
                    int((HDMI_resolution[1] - (self.factor * self.resolution[1] * 2 - 1)) / 2)]
        self.pad_width = int(DMD_resolution[1] * 2 - np.kron(
            np.zeros(self.resolution), np.ones((self.factor, self.factor * 2))).shape[0])

        if self.method == 'Fourier' or self.method.split('_')[0] == 'Fourier':  # Fourier basis patterns
            n = 3  # 3 or 4 step (not many reasons to choose 4)
            self.matrix_all = np.zeros([int(pixels * n), pixels])  # this is 2D
            for i in range(n):
                self.matrix_all[i::n, ...] = np.real(np.fft.ifft(
                    np.diag(np.ones(pixels)).dot(np.exp(1j * 2 * np.pi * i / n))
                ))
            # normalise between 0 and 1
            self.matrix_all = self.matrix_all - np.amin(self.matrix_all, axis=0)
            self.matrix_all = self.matrix_all / np.amax(self.matrix_all, axis=0)
            if self.method == 'Fourier_binary':  # binarised version of Fourier basis pattern
                # use upsampling and Floyd-Steinberg error diffusion dithering to generate binary Fourier basis patterns
                # dx.doi.org/10.1038/s41598-017-12228-3
                raise NotImplementedError(
                    f"'{self.method}' method is not working yet. Use '{self.method.split('_')[0]}' method.")

            self.matrix_all = self.matrix_all.reshape([-1, *self.resolution])  # make 3D
            # self.matrix_all = self.matrix_all[:int(self.matrix_all.shape[0] / 2), ...]  # discard ????????#todo
        else:
            if self.method.split('_')[0] == 'Hadamard':
                matrix_both = np.zeros([pixels * 2, pixels])
                self.matrix = hadamard(pixels)  # 'Hadamard_Natural': Hadamard matrix
                if self.method == 'Hadamard_Walsh':  # Walsh ordering Hadamard matrix
                    self.matrix = Walsh(self.resolution[0], self.matrix)
            elif self.method == 'Random':  # random 50/50 matrix
                self.matrix = random_masks(pixels, self.frac, seed=self.seed)
                matrix_both = np.zeros([int(pixels * self.frac) * 2, pixels])
                self.frac = 1
            else:
                raise NotImplementedError(f"there is no '{self.method}' matrix")

            matrix_both[0::2, ...] = (1 + self.matrix) / 2  # 1 & 0
            matrix_both[1::2, ...] = (1 - self.matrix) / 2  # 0 & 1

            self.matrix_all = matrix_both.reshape([-1, *self.resolution])

        # set up fullscreen cv2 output (masks on the DMD)
        self.window = set_up_mask_output()

        # set up the multimeter
        plc = 1 / 50
        self.rm, self.multimeter = set_up_multimeter()
        # prepare the multimeter for a run of measurements
        self.reset_multimeter()
        # set the measurement time
        self.integration_time = plc * self.xplc  # in seconds
        self.measurements_time = self.integration_time * self.minimum_measurements_per_mask  # in seconds

    def take_picture(self, pause_time=15):
        # reconstruct image from measurements and masks

        measurements_p_and_m = self.measure(pause_time)
        # measurements_p_and_m = np.loadtxt("outputs/measurements.txt")

        if self.method == 'Random':
            # save the measurement matrix (and its seed so it can be reconstructed in python), as the matrix is random
            np.savetxt(f"outputs/meas_matrix_{str(self.seed).zfill(3)}.txt", self.matrix, fmt='%d')
            return
        elif self.method.split('_')[0] == 'Fourier':
            measurements = measurements_p_and_m
            # measurements[:int(measurements_p_and_m.shape[0] / 2)] = 0
        else:
            measurements = measurements_p_and_m[0::2] - measurements_p_and_m[1::2]  # differential measurement
        image = self.reconstruct(measurements)
        plt.imsave(f"outputs/SPC_image_{self.resolution[0]}_{self.xplc}"
                   f"_{self.minimum_measurements_per_mask}_{self.frac}_{self.method}.png",
                   image, cmap=plt.get_cmap('gray'))  # save image
        plt.imsave(f"outputs/SPC_image_{self.resolution[0]}_{self.xplc}"
                   f"_{self.minimum_measurements_per_mask}_{self.frac}_{self.method}_upscaled.png",
                   np.kron(image, np.ones((10, 10))), cmap=plt.get_cmap('gray'))  # 10x integer upscaled image

    def reconstruct(self, measurements):
        # returns a 2D array with integer values 0 to 255, which is the image reconstructed from the measurements
        if self.method.split('_')[0] == 'Hadamard':
            image = self.matrix @ measurements
        elif self.method.split('_')[0] == 'Fourier':
            m1 = measurements[1::3]
            m2 = measurements[2::3]
            image = np.real(np.fft.ifft((measurements[0::3].dot(2) - m1 - m2) + np.sqrt(3) * 1j * (m1 - m2)))  # 3-step
            # image = np.real(np.fft.ifft((measurements[0::4] - measurements[2::4]) +
            #                             1j * (measurements[1::4] - measurements[3::4])))  # 4-step
        else:
            image = np.linalg.solve(self.matrix, measurements)  # attempt at a catch-all reconstruct (no better than @)
        return np.uint8(np.rint(
            ((image - np.amin(image)) / (np.amax(image) - np.amin(image))).reshape(self.resolution) * 255
        ))  # [..., ::-1]

    def measure(self, pause_time):
        measurements = np.zeros([self.matrix_all.shape[0]])
        deviations = np.zeros([self.matrix_all.shape[0]])
        numbers = np.zeros([self.matrix_all.shape[0]])
        # times = np.zeros([self.matrix_all.shape[0]])

        # the first cv2 display takes a while to set up, and this was causing problems with the measurements
        cv2.imshow(self.window, cv2.imread("originals/splash.png"))  # first cv2 display
        cv2.waitKey(17)
        time.sleep(pause_time)

        iterations = 2 * int(self.matrix_all.shape[0] * self.frac / 2)

        if self.method.split('_')[0] == 'Hadamard':
            i = 2  # ignoring problematic 0 and 1
        else:
            i = 0

        if self.method.split('_')[0] == 'Fourier':
            next_mask = np.uint8(np.rint(self.make_fourier_mask(i) * 255))
        else:
            next_mask = np.uint8(np.rint(self.reshape_mask(self.matrix_all[i, ...]) * 255))  # todo NOT +1
        while i < iterations:
            cv2.imshow(self.window, next_mask)
            # show the window (for the display time)
            cv2.waitKey(17)
            time.sleep(3 * (1 / 60))  # (1/60)s, to make sure this mask is displaying before we take any measurements
            # todo 2 * (1/ 60)? How low can we push this

            # tell the multimeter to wait for a trigger (which happens instantly since the trigger is set to immediate)
            self.multimeter.write('INITiate')
            start = time.time()
            if i < self.matrix_all.shape[0] - 1:
                # next_mask = np.uint8(self.reshape_mask(self.matrix_all[6, ...]) * 255)
                if self.method.split('_')[0] == 'Fourier':
                    # start = time.time()
                    next_mask = np.uint8(np.rint(self.make_fourier_mask(i) * 255))
                    # print(time.time() - start)
                else:
                    next_mask = np.uint8(np.rint(self.reshape_mask(self.matrix_all[i + 1, ...]) * 255))
                # next_mask = np.uint8(np.ones([608, 684]) * /255)
            sleep_time = self.measurements_time - (time.time() - start)
            if sleep_time > 0:  # wait for the remaining amount of time to take a multiple measurements
                # print(f"slept for {sleep_time}s on mask {i}")
                time.sleep(sleep_time)
            else:
                print(f"did not sleep on mask {i}")
            try:
                # fetch the data from the instruments internal storage and format them
                buffer = np.array(self.multimeter.query('FETCh?').split(';')[0].split(','), dtype=float)
                deviations[i] = np.std(buffer)
                measurements[i] = np.mean(buffer)
                numbers[i] = len(buffer)  # int(np.shape(buffer)[0])
                # times[i] = time.time() - start
                # while len(buffer) < self.minimum_measurements_per_mask:
                # while deviations[i] > 0.01 * measurements[i] or len(buffer) < self.minimum_measurements_per_mask:
                while deviations[i] > 5e-6 or len(buffer) < self.minimum_measurements_per_mask:
                    # todo change threshold
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
        np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/deviations.txt", deviations)
        # np.savetxt("outputs/times.txt", times)
        np.savetxt("outputs/measurements.txt", measurements)  # fmt='%.5e'????? precision
        return measurements

    def reshape_mask(self, mask):
        # reshape by: integer upscale, diamond grid correction, rotate 90, pad to be the correct HDMI shape
        return np.pad(np.rot90(my_45(np.kron(mask, np.ones((self.factor, self.factor)))), axes=(0, 1)),
                      ((self.pad[0], self.pad[0]), (self.pad[1], self.pad[1] + 1)))

    def make_fourier_mask(self, i):
        # points = np.array((range(self.resolution[0]), range(self.resolution[1]))).T
        # xi = np.array((np.arange(self.resolution[0] * self.factor) / self.resolution[0],
        #                np.arange(self.resolution[1] * self.factor) / self.resolution[1]))
        # mask = griddata(points, self.matrix_all[i, ...], xi, method='nearest')
        interp = interp2d(range(self.resolution[0]), range(self.resolution[1]), self.matrix_all[i, ...], kind='linear')
        # copy=False
        # {‘linear’, ‘cubic’, ‘quintic’}
        mask = interp(np.arange(self.resolution[0] * self.factor) / self.factor,
                      np.arange(self.resolution[1] * self.factor) / self.factor, assume_sorted=True)
        mask = np.pad(my_45(mask), ((self.pad[0], self.pad[0]), (self.pad[1], self.pad[1] + 1)))
        mask = np.where(mask < 0, 0, mask)
        return np.where(mask > 1, 1, mask)

    def reset_multimeter(self):
        # set the multimeter up to take (and store) infinite measurements as fast as possible when called to INITiate

        # tell the multimeter to clear the buffer and reset
        self.multimeter.write('*RST')

        # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
        self.multimeter.write('CONFigure:VOLTage:DC 0.1, MIN')  # 0.1V = 100mV

        # set the number of PLC for each integration - i.e. measurement time
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