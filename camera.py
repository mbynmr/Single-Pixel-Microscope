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
        pixels = np.asarray(self.resolution).prod()  # e.g. 128*128
        self.DMD_resolution = [608, 684]
        self.factor = int(self.DMD_resolution[0] / self.resolution[0])  # todo floor, and pad vertically to stop rounding
        self.pad_width = int((self.DMD_resolution[1] * 2 -
                              np.kron(np.zeros(self.resolution), np.ones((self.factor, self.factor * 2))).shape[0]))
        # 760

        # Hadamard matrix
        self.hadamard_mat = hadamard(pixels)
        self.hadamard_all = np.zeros([pixels * 2, pixels])
        self.hadamard_all[0::2, ...] = (1 + self.hadamard_mat) / 2  # 1 & 0
        self.hadamard_all[1::2, ...] = (1 - self.hadamard_mat) / 2  # 0 & 1

        # set up fullscreen cv2 output (masks on the DMD)

        self.display_time = 200 * 1e-3  # in seconds = 3 * (1 / 60)
        self.display_time_ms = int(self.display_time * 1e3)  # in milliseconds
        # self.display_time_ms = int(1000 / 60)  # in milliseconds
        self.window = set_up_mask_output()

        # set up the multimeter
        self.rm, self.multimeter = set_up_multimeter()
        # prepare the multimeter for a run of measurements
        self.reset_multimeter()

    def take_picture(self):
        measurements_p_and_m = self.measure()
        measurements = measurements_p_and_m[0::2, ...] - measurements_p_and_m[1::2, ...]  # differential measurement

        # reconstruct image from measurements and masks
        image = self.reconstruct(measurements)
        plt.imsave(f"outputs/output{0}.png", image, cmap=plt.get_cmap('gray'))

    def reconstruct(self, measurements):
        image = self.hadamard_mat @ measurements
        return np.uint8(image.reshape(self.resolution) * 255)  # todo formatting

    def measure(self):
        measurements = np.zeros([self.hadamard_all.shape[0]])
        start = time.time()
        for i, mask in enumerate(self.hadamard_all[:20, ...]):  # todo first 20 only!
            # convert the plus & minus hadamard matrixes into the correct images to be displayed on the DMD
            # turn 32x32 masks into 608x1216 by integer scaling
            mask_show = np.kron(mask.reshape(self.resolution), np.ones((self.factor, self.factor * 2)))
            # pad the rectangular mask with zeros on both sides, so the on the DMD it appears as a roughly centred square
            mask_show = np.pad(mask_show, ((0, 0), (self.pad_width - 400, 400)))  # 608x1976  # todo 400
            # the DMD rotates the image 90 degrees clockwise, so we need to do the opposite
            mask_show = np.rot90(mask_show, axes=(1, 0))  # 1976x608

            # convert to uint8 and show on the window
            cv2.imshow(self.window, np.uint8(mask_show * 255))
            # show the window for the display time
            cv2.waitKey(self.display_time_ms)

            # the first <100ms that the mask is displayed is spent waiting
            # print((time.time() - start) % self.display_time)
            try:
                time.sleep(100e-3 - (time.time() - start) % self.display_time)  # todo 100e-3
            except ValueError:
                # if we miss this frame, wait until the next
                print("Too slow! Waiting until the next frame...")
                time.sleep(self.display_time + 100e-3 - (time.time() - start) % self.display_time)

            # the next ~400ms is spent taking measurements
            # tell the multimeter to wait for a trigger (which happens instantly since the trigger is set to immediate)
            self.multimeter.write('INITiate')
            time.sleep(0.001)

            # print(self.my_instrument.query('DATA: POINts?'))
            # if int(self.my_instrument.query('DATA: POINts?')) > 900:
            if len(self.multimeter.query('FETCh?').split(',')) > 900:
                print("resetting multimeter")
                self.reset_multimeter()
            # time.sleep(self.display_time - (time.time() - start) % self.display_time)
            for _ in range(5):
                buffer = self.multimeter.query('FETCh?')
                print(buffer)
                print(buffer.split(';')[0].split(','))
            print("once")
            # Fetch the data from the instruments internal storage, then take the average of them
            # measurements[i] = np.mean(np.array(self.my_instrument.query('FETCh?').split(','), dtype=float))
            # print(self.my_instrument.query('FETCh?'))

        return measurements

    def reset_multimeter(self):
        # set the multimeter up to take (and store) infinite measurements as fast as possible when called to INITiate

        # tell the multimeter to clear the buffer and reset
        self.multimeter.write('*RST')

        # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
        self.multimeter.write('CONFigure:VOLTage:DC 10, MIN')

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
        self.multimeter.write('*RST')  # tell the multimeter to reset
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
