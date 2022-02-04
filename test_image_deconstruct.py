from scipy.linalg import hadamard
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np


def deconstruct_hadamard(resolution, file):
    # takes an image and deconstructs it into a series of measurements and the corresponding Hadamard masks

    try:  # todo need a better way of making the input image always greyscale, rather than this
        original = rgb2gray(rgba2rgb(plt.imread(f"originals/{file}")))
    except ValueError:
        try:
            original = rgb2gray(plt.imread(f"originals/{file}"))
        except ValueError:
            original = plt.imread(f"originals/{file}")

    # resize to reconstruction size, flatten into 1D array
    original_resized = resize(original, resolution).flatten()

    # Hadamard matrix
    hadamard_mat = hadamard(np.asarray(resolution).prod())

    # using patterns with both -1 and 1, measurements are
    meas = hadamard_mat @ original_resized
    return meas, hadamard_mat
