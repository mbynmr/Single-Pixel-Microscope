from scipy.linalg import hadamard
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from test_image_deconstruct import deconstruct_hadamard


class Reconstructor:
    def __init__(self, resolution):
        self.resolution = np.asarray(resolution)

        # Hadamard matrix
        self.hadamard_mat = hadamard(np.asarray(self.resolution).prod())

    def reconstruct(self, measurements, mask_indexes):
        measurements_arranged = np.zeros(self.resolution.prod())  # todo is this the correct size?
        for i, mask_index in enumerate(mask_indexes):
            measurements_arranged[mask_index] = measurements[i]
        # todo the above could be redundant if we measure by always setting the measurements matrix to the right size

        image = (self.hadamard_mat @ measurements_arranged).reshape(self.resolution)
        return image

    def save_image(self, measurements, mask_indexes):
        plt.imsave('outputs/output.png', self.reconstruct(measurements, mask_indexes), cmap=plt.get_cmap('gray'))


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


def find_nth_best_masks(resolution, measurements, sampling_ratio):
    threshold = np.sort(abs(measurements))[-int(np.asarray(resolution).prod() * sampling_ratio)]
    indexes = (abs(measurements) >= threshold)
    # indexes = np.nonzero((abs(measurements) >= threshold))
    return indexes


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
