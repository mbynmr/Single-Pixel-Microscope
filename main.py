import numpy as np

from test_image_deconstruct import deconstruct_hadamard
from image_reconstruct import Reconstructor, reconstruct, reconstruct_with_other_images_best_masks


# image resizer
#  ffmpeg -i mario.png -vf scale=128:-1 mario128.png

# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm
#  assumes square image dimensions (128x128, 64x64, etc)
#  converts to greyscale

def main():
    resolution = [128, 128]  # resolution of reconstruction
    file = 'mario.png'
    # file = 'IMG-0943.JPG'  # dark background
    # file = 'IMG-0947.JPG'  # light background
    # file = 'IMG-0948.JPG'  # light background

    # measurements, masks = deconstruct_hadamard(resolution, file)
    # reconstruct(resolution, measurements, masks)

    # file1 = 'mario.png'
    # file2 = 'IMG-0944.JPG'
    # reconstruct_with_other_images_best_masks(resolution, file1, file2)

    r = Reconstructor(resolution)
    # test the reconstructor using the deconstructor
    measurements, masks = deconstruct_hadamard(resolution, file)
    indexes = list(range(int(np.asarray(resolution).prod())))
    for i in range(len(indexes)):
        n = 10
        if np.random.randint(low=0, high=n) == 0:  # randomly remove (1/n) of the measurements
            measurements[i] = 0
            indexes.remove(i)
    r.save_image(measurements, indexes)


if __name__ == '__main__':
    main()
