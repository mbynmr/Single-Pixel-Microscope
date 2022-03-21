from scipy.linalg import hadamard
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def deconstruct_general(resolution, image, matrix):
    # takes image and returns the series of deconstructed measurements dependent on the mask matrix

    original = plt.imread(f"originals/{image}")
    if original.shape[-1] == 4:
        original = rgba2rgb(original)
    if original.shape[-1] == 3:
        original = rgb2gray(original)

    return matrix @ resize(original, resolution).flatten()


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


def test_image():
    image = plt.imread("originals/mario.png")
    if image.shape[-1] == 4:
        image = rgba2rgb(image)
    plt.imsave("outputs/mario_rgb.png", resize(image, (684, 608)))
    if image.shape[-1] == 3:
        image = rgb2gray(image)
        plt.imsave("outputs/mario_grey.png", resize(image, (684, 608)), cmap=plt.get_cmap('gray'))


def test_image_two():
    colorImage = Image.open("outputs/mario_rgb.png")
    imageWithColorPalette = colorImage.convert("YCbCr", palette=Image.ADAPTIVE, colors=24)
    # imageWithColorPalette.show()
    imageWithColorPalette.save("outputs/mario_rgb_24_bit.jpg")
