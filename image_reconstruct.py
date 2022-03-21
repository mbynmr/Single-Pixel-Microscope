from scipy.linalg import hadamard
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from test_image_deconstruct import deconstruct_general, deconstruct_hadamard


class Reconstructor:
    """
    Reconstructs an image from measurements corresponding to a sequence of Hadamard masks
    """

    def __init__(self, resolution, method='Hadamard'):
        self.resolution = np.asarray(resolution)  # e.g. [128, 128]
        self.pixels = self.resolution.prod()  # e.g. 128*128
        self.hadamard_bool = False

        # mask matrix
        if method == 'Hadamard':  # Hadamard matrix
            # hadamard_mat = hadamard(self.pixels)           # 1 & -1
            # self.hadamard_plus = (1 + self.hadamard_mat) / 2    # 1 & 0
            # self.hadamard_minus = (1 - self.hadamard_mat) / 2   # 0 & 1
            self.matrix = hadamard(self.pixels)
            self.hadamard_bool = True
        elif method == 'random':  # random matrix
            self.matrix = np.random.randint(low=0, high=2, size=[self.pixels, self.pixels])
            # self.matrix = np.random.randint(low=-1, high=2, size=[self.pixels, self.pixels])  # todo -1, 0, 1?
        else:
            raise NotImplementedError(f"there is no '{method}' method")

    def measure(self, file, do_return=None):
        measurements = deconstruct_general(self.resolution, file, self.matrix)
        if do_return is None:  # either return the measurements or save them to a file
            return measurements
        else:
            raise NotImplementedError("work in progress")
            # np.savetxt('outputs/measurement.txt', measurements, '%.5e', ',', '\n')
            # check that the size of the file won't be too large (I have a feeling it will be on the order of 1GB)

    def reconstruct(self, measurements):
        if self.hadamard_bool:  # faster method (only works for Hadamard masks, not random masks)
            return (self.matrix @ measurements).reshape(self.resolution)
        elif np.any(measurements == 0):
            self.matrix = self.matrix[..., measurements != 0]
            measurements = measurements[measurements != 0]
            answer = np.asarray(1)
            # todo l1 minimisation
            #  use pdf lecture notes on compressed sensing
            #  find out how to do l1-norm minimisation in python: https://stackoverflow.com/questions/58584127
            return answer.reshape(self.resolution)
        else:
            return np.linalg.solve(self.matrix, measurements).reshape(self.resolution)
        # the differences between np.linalg.solve(m1, m2) and m1 @ m2 are:
        #  np.linalg.solve is slower than @ (around 10x for 32x32 and around 15x for 64x64)
        #  @ doesn't work for reconstructing using random masks

    def save_image(self, measurements):
        # save without ever overwriting a previous output image
        n = int(0)
        try:
            while True:
                plt.imread(f"outputs/output{n}.png")
                n += 1
        except FileNotFoundError:
            plt.imsave(f"outputs/output{n}.png", self.reconstruct(measurements), cmap=plt.get_cmap('gray'))

    def undersample(self, measurements, method='last', portion=0.9):
        if method == 'last':
            self.matrix[..., int(measurements.shape[0] * portion):] = 0
            self.matrix[0, int(measurements.shape[0] * portion):] = 1
            measurements[int(measurements.shape[0] * portion):] = 0
        elif method == 'first':
            self.matrix[..., :int(measurements.shape[0] * portion)] = 0
            measurements[:int(measurements.shape[0] * portion)] = 0
        elif method == 'middle':
            self.matrix[..., int(measurements.shape[0] * (0.5 - (1 - portion) / 2)):
                             int(measurements.shape[0] * (0.5 + (1 - portion) / 2))] = 0
            measurements[int(measurements.shape[0] * (0.5 - (1 - portion) / 2)):
                         int(measurements.shape[0] * (0.5 + (1 - portion) / 2))] = 0
        elif method == 'random':
            random_indexes = np.random.random(measurements.shape) < portion  # the same random indexes for mat & meas
            self.matrix = self.matrix[..., random_indexes]
            return np.where(random_indexes, measurements, 0)
        else:
            raise NotImplementedError(f"there is no '{method}' method")
        return measurements


def add_noise(measurements, multiplier=1e-2, method='normal'):
    # multiplier is found from our "worst case" noise from real measurements
    std = multiplier * measurements  # standard deviation found from multiplier

    if method == 'normal':
        return measurements + np.multiply(std, np.random.randn(*measurements.shape))  # normal noise
    elif method == 'uniform':
        return measurements + np.multiply(std, 1 - 2 * np.random.random(*measurements.shape))  # uniform noise
    else:
        raise NotImplementedError(f"there is no '{method}' method, choose random or uniform noise")


def find_nth_best_masks(resolution, measurements, sampling_ratio):
    threshold = np.sort(abs(measurements))[-int(np.asarray(resolution).prod() * sampling_ratio)]
    indexes = (abs(measurements) >= threshold)
    # indexes = np.nonzero((abs(measurements) >= threshold))
    return indexes


def reconstruct(resolution, measurements, hadamard_mat):
    # reconstructs an image from measurements and the corresponding Hadamard masks
    # sampling ratios
    sampling_ratio_list = [1, 0.9, 0.5, 0.25, 0.1, 0.05]

    fig, axes = plt.subplots(2, int(len(sampling_ratio_list) / 2), figsize=(12, 8))

    for k, sampling_ratio in tqdm(enumerate(sampling_ratio_list)):
        meas = measurements  # reset meas back to the input measurements on every loop
        if sampling_ratio < 1:
            best = True  # comment out either True or False to get either the best or random sampling% measurements
            # best = False
            if best:
                # selecting highest values
                threshold = np.sort(abs(meas))[-int(np.asarray(resolution).prod() * sampling_ratio)]
                meas[abs(meas) < threshold] = 0
                indexes = np.nonzero((abs(meas) < threshold))
            else:
                random_indexes = np.random.choice(range(np.prod(np.shape(meas))),
                                                  int(np.asarray(resolution).prod() * (1 - sampling_ratio)),
                                                  replace=False)
                meas[random_indexes] = 0

        # reconstruction by weighted sum/inverse hadamard transform
        reconstructed = (hadamard_mat @ meas).reshape(resolution)
        # reconstructed =  np.linalg.solve(hadamard_mat, meas).reshape(resolution)

        # plotting each reconstructed image
        j = k % 3
        i = (k - j) % 2
        axes[i][j].imshow(reconstructed, cmap=plt.get_cmap('gray'))
        axes[i][j].set_axis_off()
        axes[i][j].set_title(f"{sampling_ratio * 100}% sampling ratio")

    plt.show()  # may not be required for some interpreters


def reconstruct_with_other_images_best_masks(resolution, file1, file2):
    # compares reconstructing with file1's best masks to reconstructing with file2's best masks
    # sampling ratios
    # sampling_ratio_list = [1, 0.9, 0.5, 0.25]
    sampling_ratio_list = [1, 0.5, 0.25, 0.1]

    fig, axes = plt.subplots(len(sampling_ratio_list), 3)

    measurements_other, masks_other = deconstruct_hadamard(resolution, file1)
    measurements, masks = deconstruct_hadamard(resolution, file2)

    for k, sampling_ratio in tqdm(enumerate(sampling_ratio_list)):
        indexes_other = find_nth_best_masks(resolution, measurements_other, sampling_ratio)
        indexes = find_nth_best_masks(resolution, measurements, sampling_ratio)

        # this image's best masks
        reconstructed0 = (masks @ (measurements * indexes)).reshape(resolution)
        axes[k][0].imshow(reconstructed0, cmap=plt.get_cmap('gray'))
        axes[k][0].set_title(f"this image's best masks, {sampling_ratio * 100}%")
        axes[k][0].set_axis_off()

        # similar image's best masks
        if sampling_ratio == 1:
            reconstructed1 = (masks_other @ (measurements_other * indexes_other)).reshape(resolution)
            axes[k][1].set_title(f"other image")
        else:
            reconstructed1 = (masks @ (measurements * indexes_other)).reshape(resolution)
            axes[k][1].set_title(f"other image's best masks, {sampling_ratio * 100}%")
        axes[k][1].imshow(reconstructed1, cmap=plt.get_cmap('gray'))
        axes[k][1].set_axis_off()

        # random masks
        indexes_random = np.random.choice(range(np.prod(np.shape(measurements))),
                                          int(np.asarray(resolution).prod() * (1 - sampling_ratio)), replace=False)
        meas = measurements
        meas[indexes_random] = 0
        reconstructed2 = (masks @ meas).reshape(resolution)
        axes[k][2].imshow(reconstructed2, cmap=plt.get_cmap('gray'))
        axes[k][2].set_title(f"random masks, {sampling_ratio * 100}%")
        axes[k][2].set_axis_off()

    plt.show()  # may not be required for some interpreters
