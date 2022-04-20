# from notify_run import Notify

from test_stuff import noise_and_undersampling, rotating_masks, fourier_masks
from image_reconstruct import Reconstructor
# from camera import Camera
import numpy as np
import matplotlib.pyplot as plt

# image resizer: ffmpeg -i mario.png -vf scale=128:-1 mario128.png
# image type converter: ffmpeg -i image.webp image.gif
# xmodmap -e "keycode 49 = backslash"

# very helpful: https://github.com/cbasedlf/single_pixel_demo/blob/master/sp_demo.py
# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm opencv-python pyvisa-py notify-run graycode
#  always square image dimensions (128x128, 64x64, etc) - is it possible to have non-square dimensions? I think so
#  converts input (test) images to greyscale
#  use images as masks? may be useful for AI cases "is this a dog?" compressed sensing could do it very fast
#  deconstruct 2 images and compare the measurements to tell how alike images are?

# todo Note to try: reconstruct random masks with linalg solve UNDERSAMPLED by:
#  repeating measurements and their masks, rather than setting them to zero.
#  e.g. reconstruct with masks 1, 2, 3, 4, 1, 2, 3, 4 instead of 8 distinct masks. Surely it'll work?


def not_main():
    power = int(7)  # 4: 16x16, 5: 32x32, 6: 64x64, 7: 128x128, 8: 256x256

    fourier_masks()
    return

    # from HadamardOrdering import random_masks
    # masks = random_masks(pixels=np.asarray([2 ** power, 2 ** power]).prod(), frac=1, seed=10)
    # print(masks[0:4, 0:4])
    # return

    # r = Reconstructor(resolution=[2 ** power, 2 ** power], method=method.split('_')[0])
    # measurements = np.loadtxt("outputs/measurements.txt")
    # measurements = measurements[0::2] - measurements[1::2]  # differential measurement
    # image = r.reconstruct(measurements)
    # image = image - np.amin(image)
    # image = image + np.amax(image)
    # plt.imsave("outputs/reconstruct.png", image, cmap=plt.get_cmap('gray'))
    # return


def main():
    power = int(4)  # 4: 16x16, 5: 32x32, 6: 64x64, 7: 128x128, 8: 256x256

    xplc_index = int(2)  # 0 to 3: [0.02, 0.1, 1, 10]
    measurements_per_mask = int(3)

    fraction = 1
    method = 'Hadamard_Natural'
    # method = 'Hadamard_Walsh'
    # method = 'Random'
    # method = 'Fourier'
    # method = 'Fourier_binary'

    try:
        resolution = [2 ** power, 2 ** power]
        print(f"{resolution = }")
        xplc = [0.02, 0.1, 1, 10]  # these are the only options for the multimeter

        # c = Camera(resolution, xplc[xplc_index], measurements_per_mask, fraction, method)
        # c.take_picture(pause_time=5)  # input pause time in seconds before the masks are shown
        # c.close()
    finally:
        # notify = Notify()
        # notify.send('Finished')
        # Notify().send('Finished')
        print("done")


if __name__ == '__main__':
    # main()
    not_main()
