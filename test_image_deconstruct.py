from scipy.linalg import hadamard
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np


def deconstruct_hadamard(resolution, file):
    # takes an image and deconstructs it into a series of measurements and the corresponding Hadamard masks

    original = plt.imread(f"originals/{file}")
    if original.shape[-1] == 4:
        original = rgba2rgb(original)
    if original.shape[-1] == 3:
        original = rgb2gray(original)

    # resize to reconstruction size, flatten into 1D array
    original_resized = resize(original, resolution).flatten()

    # Hadamard matrix
    hadamard_mat = hadamard(np.asarray(resolution).prod())

    # using patterns with both -1 and 1, measurements are
    meas = hadamard_mat @ original_resized
    return meas, hadamard_mat
