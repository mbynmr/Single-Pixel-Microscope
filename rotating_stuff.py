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
    # i = dim 0 = up/down
    # j = dim 1 = left/right

    n = mask.shape[0]  # mask is n x n
    my_mask = np.zeros([2 * n - 1, n])  # my_mask is (2n - 1) x n

    # ---- first rows ----
    for diagonal in range(n - 1):
        row = np.zeros([n])
        for j in range(diagonal):
            i = diagonal - j  # condition: i + j = diagonal
            if i < n and -j < n:
                if diagonal % 2 == 0:  # even
                    # pad with an extra zero on the left
                    row[j - diagonal] = mask[i, -j]
                else:  # odd
                    # pad equally on the left and right
                    row[j - diagonal + 1] = mask[i, -j]
        my_mask[diagonal, :] = row

    # ---- middle row ----
    row = np.zeros([n])
    for j in range(n):
        i = j
        row[i] = mask[i, j]
    my_mask[n - 1, :] = row

    # ---- last  rows ----
    for d in range(n - 1):
        diagonal = d + n
        row = np.zeros([n])
        for j in range(diagonal):  # todo completely copied from the first n
            i = diagonal - j  # condition: i + j = diagonal
            if i < n and j < n:
                if diagonal % 2 == 0:  # even
                    # pad with an extra zero on the left
                    row[j] = mask[i, -j]
                else:  # odd
                    # pad equally on the left and right
                    row[j] = mask[i, -j]
        # i = 2 * n - 1 - diagonal  # condition: i + j = 2n - 1 - diagonal
        my_mask[diagonal, :] = row
    return my_mask
