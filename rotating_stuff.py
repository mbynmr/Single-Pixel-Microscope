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
        mask[0, 0] = 1#todo remove
        mask[1, 0] = 2
        mask[2, 0] = 3
        mask[3, 0] = 4
        # mask[0, 0] = 1
        # mask[0, 1] = 2
        # mask[0, 2] = 3
        # mask[0, 3] = 4
        mask = mask + 0.25 * np.diag(np.ones(mask.shape[0]))
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
    n = mask.shape[0]  # mask is n x n
    my_mask = np.zeros([2 * n - 1, n])  # my_mask is (2n - 1) x n

    # ---- first rows ----
    for diagonal in range(n):  # this overwrites row -1, but as long as it is done first, that's not a problem
        for i in range(diagonal):
            my_mask[diagonal - 1, i + 1 + int((n - 1 - diagonal) / 2)] = mask[i, n + i - diagonal]

    # ---- middle row ----
    # diagonal = n
    for i in range(n):
        my_mask[n - 1, i] = mask[i, i]

    # ---- last  rows ----
    for d in range(n - 1):
        # todo check we don't get diagonal = n and overwrite the middle!
        diagonal = 2 * n - (d + 1)  # working backward from 2 * n - 1 to n
        row = np.zeros([n])
        for i in range(d + 1):  # 2 * n - diagonal = n - (d + 1)
            j = d - i  # condition: i + j = d
            # (i) 2,2,3
            # d=0,1,2
            # 2 + int(d / 2)
            row[-i + int((n + d) / 2)] = mask[n - 1 - i, j]  # works for 4x4
            # row[-i + 2 + int(d / 2)] = mask[n - 1 - i, j]  # works for 4x4
        my_mask[diagonal - 1, :] = row
    # i = dim 0 = up/down. top = 0, bottom = 3
    # j = dim 1 = left/right. left = 0, right = 3
    return my_mask
