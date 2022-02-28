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

        # Hadamard matrix
        self.hadamard_mat = hadamard(pixels)
        hadamard_plus = (1 + self.hadamard_mat) / 2  # 1 & 0
        hadamard_minus = (1 - self.hadamard_mat) / 2  # 0 & 1
        self.hadamard_all = np.zeros([pixels * 2, pixels])
        self.hadamard_all[0::2, ...] = hadamard_plus
        self.hadamard_all[1::2, ...] = hadamard_minus

        # setup fullscreen cv2 output
        self.window = 'output_window'
        self.setup_output()

        # setup the multimeter ready to take measurements
        self.rm = []
        self.my_instrument = []
        self.setup_input()

    def take_picture(self):
        measurements_p_and_m = self.measure()
        measurements_plus = measurements_p_and_m[0:2, ...]
        measurements_minus = measurements_p_and_m[1:2, ...]
        measurements = measurements_plus - measurements_minus  # differential measurement

        # reconstruct image from measurements and masks
        image = self.reconstruct(measurements)
        plt.imsave(f"outputs/output{0}.png", image, cmap=plt.get_cmap('gray'))

    def reconstruct(self, measurements):
        image = self.hadamard_mat @ measurements
        return image.reshape(self.resolution)

    def measure(self):
        measurements = np.zeros([self.hadamard_all.shape[0]])
        for i, mask in enumerate(self.hadamard_all):
            # convert the plus & minus hadamard matrixes into the correct images to be displayed on the DMD
            mask_show = np.kron(mask.reshape(self.resolution), np.ones((self.factor, self.factor * 2)))
            pad_width = int((self.DMD_resolution[1] * 2 - mask_show.shape[0]))  # 760  # todo work out pad_width once

            # pad the mask with zeros either side of the rectangle, so the on the DMD
            mask_show = np.pad(mask_show, ((0, 0), (pad_width - 400, 400)))  # 608x1976  # todo -400, 400 centres it

            # the DMD rotates the image 90 degrees (), so we need to do the opposite #todo(clockwise/anticlockwise)
            mask_show = np.rot90(np.rot90(np.rot90(mask_show)))  # 1976x608  # todo rotate 90 the other way once

            # convert to uint8 and show on the window
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # todo does it need RGB input?
            cv2.imshow(self.window, np.uint8(mask_show * 255))
            # pause for (time)#todo time is rounded!
            cv2.waitKey(17)  # in milliseconds ~=1000/60

            # todo timing measurements and display
            num_of_measures = int(1e5)
            all_measurements = np.ones([num_of_measures]) * np.nan
            time.sleep(1/1000)
            start = time.time
            j = 0
            while float(time.time) < float(start) + 15:
                # Fetch the data from the instruments internal storage to the buffer, and read in the data.
                all_measurements[j] = float(self.my_instrument.query('FETCh?'))
                j += 1
            # average measurements
            measurements[i] = np.nanmean(all_measurements)
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

        # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
        self.my_instrument.write('CONFigure:VOLTage:DC 10, MIN', )
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
        self.my_instrument.clear()
        self.my_instrument.close()
        self.rm.close()
