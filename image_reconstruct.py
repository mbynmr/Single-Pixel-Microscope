from scipy.linalg import hadamard
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from test_image_deconstruct import deconstruct_hadamard


class Reconstructor:
    """Reconstructs an image from measurements corresponding to a sequence of Hadamard masks"""
    def __init__(self, resolution):
        self.resolution = np.asarray(resolution)  # e.g. [128, 128]
        self.pixels = self.resolution.prod()  # e.g. 128*128

        # Hadamard matrix
        self.hadamard_mat = hadamard(self.pixels)           # 1 & -1
        self.hadamard_plus = (1 + self.hadamard_mat) / 2    # 1 & 0
        self.hadamard_minus = (1 - self.hadamard_mat) / 2   # 0 & 1

    def measure(self, do_return=True):
        measurements = np.zeros([self.pixels])
        # we can either skip elements, or rewrite the loop to only be over certain measurements if we want to have
        # undersampled measurements - therefore those values of the measurements array will stay as zero
        for i in tqdm(range(self.pixels)):
            # here, we have 2D masks to display on the DMD and a measurements array to store the measured value in
            # talk to the DMD with code from https://github.com/csi-dcsc/Pycrafter6500
            mask_plus = self.hadamard_plus[i, ...].reshape(self.resolution)
            mask_minus = self.hadamard_minus[i, ...].reshape(self.resolution)
            measurements[i] = np.abs(np.random.normal(0, 10))  # todo placeholder measurement
        if do_return:  # either return the measurements or save them to a file
            return measurements
        else:
            print("work in progress")
            # np.savetxt('outputs/measurement.txt', measurements, '%.5e', ',', '\n')
            # check that the size of the file won't be too large (I have a feeling it will be on the order of 1GB)

    def reconstruct(self, measurements):
        if self.hadamard_mat.shape[0] == measurements.shape[0]:
            image = self.hadamard_mat @ measurements
            # image2 = np.linalg.solve(self.hadamard_mat, measurements)
            # print(np.allclose(image, image2))
            return image.reshape(self.resolution)
            # todo find out what the difference is between np.linalg.solve and @
            #  np.linalg.solve is slower than @ (around 10x for 32x32 and around 15x for 64x64)
            #  they aren't equal according to np.allclose
            #  they are very similar but a factor of 1e3 different from each other (using random measurements)
        else:
            # the length of the measurements array tells you the masks as they are always the same Hadamard masks
            raise ValueError("The resolution was different to expected")
            # rather than raising an error, we can write code to adaptively reconstruct the image even if the mask

    def save_image(self, measurements):
        # save without ever overwriting a previous output image
        n = int(0)
        while True:
            try:
                plt.imread(f"outputs/output{n}.png")
            except FileNotFoundError:
                plt.imsave(f"outputs/output{n}.png", self.reconstruct(measurements), cmap=plt.get_cmap('gray'))
                break
            n += 1


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
