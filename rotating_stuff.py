import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate  # todo remove

from image_reconstruct import Reconstructor


def rotating_masks(resolution):
    # rotates masks by 45 degrees
    matrix = Reconstructor(resolution).matrix
    matrix_all = np.zeros([np.asarray(resolution).prod() * 2, np.asarray(resolution).prod()])
    matrix_all[0::2, ...] = (1 + matrix) / 2    # 1 & 0
    matrix_all[1::2, ...] = (1 - matrix) / 2   # 0 & 1
    for i, e in enumerate(matrix_all):
        mask = e.reshape(resolution)
        # mask[0, 0] = 1#todo remove
        # mask[1, 0] = 2
        # mask[2, 0] = 3
        # mask[3, 0] = 4
        mask[0, 0] = 1
        mask[0, 1] = 2
        mask[0, 2] = 3
        mask[0, 3] = 4
        mask_45 = rotate(mask, angle=45)
        mask_mine = my_45(mask)
        big = np.zeros([max(mask.shape[0], mask_45.shape[0], mask_mine.shape[0]),
                        mask.shape[1] + mask_45.shape[1] + mask_mine.shape[1]])
        big[:mask.shape[0], :mask.shape[1]] = mask
        big[:mask_45.shape[0], mask.shape[1]:(mask.shape[1] + mask_45.shape[1])] = mask_45
        big[:mask_mine.shape[0], (mask.shape[1] + mask_45.shape[1]):] = mask_mine
        plt.imshow(big, cmap=plt.get_cmap('gray'))
        plt.show()
    print("done")


def my_45(mask):
    # i = dim 0 = up/down. top = 0, bottom = max
    # j = dim 1 = left/right. left = 0, right = max

    n = mask.shape[0]  # mask is n x n
    my_mask = np.zeros([2 * n - 1, n])  # my_mask is (2n - 1) x n

    # ---- first rows ----
    for diagonal in range(n):
        row = np.zeros([n])
        for i in range(diagonal):
            j = diagonal - 1 - i  # condition: i + j = diagonal - 1
            row[i + 1 + int((n - 1 - diagonal) / 2)] = mask[-i, n - 1 - j]
        my_mask[diagonal - 1, :] = row

    # ---- middle row ----
    row = np.zeros([n])
    for j in range(n):
        i = j
        row[i] = mask[-i, j]
    my_mask[n - 1, :] = row

    # ---- last  rows ----
    for d in range(n):
        diagonal = d + n
        row = np.zeros([n])
        for i in range(diagonal):
            j = diagonal - 1 - i  # condition: i + j = diagonal - 1
            row[i + 1 + int((n - 1 - diagonal) / 2)] = mask[-i, n - 1 - j]
        my_mask[diagonal - 1, :] = row
    return my_mask
