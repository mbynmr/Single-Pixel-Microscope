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

        # Hadamard matrix
        self.hadamard_mat = hadamard(pixels)
        hadamard_plus = (1 + self.hadamard_mat) / 2  # 1 & 0
        hadamard_minus = (1 - self.hadamard_mat) / 2  # 0 & 1
        self.hadamard_all = np.zeros([pixels * 2, pixels])
        self.hadamard_all[0::2, ...] = hadamard_plus
        self.hadamard_all[1::2, ...] = hadamard_minus

        # setup fullscreen cv2 output
        self.window = 'output_window'
        self.display_time = 200 * 1e-3  # in seconds = 3 * (1 / 60)
        self.display_time_ms = int(self.display_time * 1e3)  # in milliseconds
        self.setup_output()#todo

        # setup the multimeter ready to take measurements
        self.rm = []
        self.my_instrument = []
        self.setup_input()

    def take_picture(self):
        measurements_p_and_m = self.measure()
        measurements = measurements_p_and_m[0::2, ...] - measurements_p_and_m[1::2, ...]  # differential measurement

        # reconstruct image from measurements and masks
        image = self.reconstruct(measurements)
        plt.imsave(f"outputs/output{0}.png", image, cmap=plt.get_cmap('gray'))

    def reconstruct(self, measurements):
        image = self.hadamard_mat @ measurements
        return image.reshape(self.resolution)

    def measure(self):
        measurements = np.zeros([self.hadamard_all.shape[0]])
        start = time.time()
        for i, mask in enumerate(self.hadamard_all[:20, ...]):
            # convert the plus & minus hadamard matrixes into the correct images to be displayed on the DMD
            mask_show = np.kron(mask.reshape(self.resolution), np.ones((self.factor, self.factor * 2)))
            # pad the mask with zeros either side of the rectangle, so the on the DMD it appears roughly centred
            mask_show = np.pad(mask_show, ((0, 0), (self.pad_width - 400, 400)))  # 608x1976  # todo 400
            # the DMD rotates the image 90 degrees clockwise, so we need to do the opposite
            mask_show = np.rot90(mask_show, axes=(1, 0))  # 1976x608

            # convert to uint8 and show on the window
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            cv2.imshow(self.window, np.uint8(mask_show * 255))
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
            self.reset_input()
            time.sleep(self.display_time - (time.time() - start) % self.display_time)
            # Fetch the data from the instruments internal storage to the buffer, then take the average of them
            print(self.my_instrument.query('FETCh?').split(','))
            measurements[i] = np.mean(np.array(self.my_instrument.query('FETCh?').split(','), dtype=float))
            # print(self.my_instrument.query('FETCh?'))

        return measurements

    def setup_output(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def setup_input(self):
        # Open an instance of the Resource manager class and assign the handle rm
        self.rm = visa.ResourceManager()
        # # Show available resources
        # print(rm.list_resources())
        # Open an instance of the USB resource class and assign the object handle my_instrument
        self.my_instrument = self.rm.open_resource('USB0::0x05E6::0x2100::1269989::INSTR')
        self.my_instrument.timeout = None  # sets no instrument timeout

    def reset_input(self):
        # tell the multimeter to clear the buffer and reset
        self.my_instrument.write('*RST')

        # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
        self.my_instrument.write('CONFigure:VOLTage:DC 10, MIN')

        # set the number of PLC for each integration - i.e. measurement time
        # print(self.my_instrument.query('SENSe:VOLTage:DC:NPLCycles?'))  # 10 PLC per measurement
        self.my_instrument.write('SENSe:VOLTage:DC:NPLCycles MINimum')
        # print(self.my_instrument.query('SENSe:VOLTage:DC:NPLCycles?'))  # 0.02 PLC per measurement

        # Set the trigger source to internal, and immediate, set the trigger delay to auto, and set the instrument to take 10
        # samples after the trigger event
        self.my_instrument.write('TRIGger:SOURce IMMediate')
        self.my_instrument.write('TRIGger:DELay:AUTO ON')
        self.my_instrument.write('SAMPle:COUNt 1')

        self.my_instrument.write('TRIG:COUN INF')  # infinite triggers

        # Set the instrument to a wait for trigger state. In the case of the Immediate internal trigger, the instrument will
        # send a trigger as soon as instrument is set to this state
        self.my_instrument.write('INITiate')

    def close(self):
        # close the cv2 fullscreen window
        cv2.destroyAllWindows()

        # clear and close the instance of the instrument, and close the resource manager
        self.my_instrument.write('*RST')  # tell the multimeter to reset
        self.my_instrument.clear()
        self.my_instrument.close()
        self.rm.close()
