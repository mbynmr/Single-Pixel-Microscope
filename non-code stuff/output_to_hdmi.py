import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hadamard
import cv2
import time

# from dmd_patterns import enlarge_pattern


def output():
    DMD_resolution = [608, 684]
    resolution = [32, 32]
    factor = int(DMD_resolution[0] / resolution[0])  # floor, and pad vertically to remove rounding (e.g. 608/64)
    pixels = np.asarray(resolution).prod()  # e.g. 128*128

    # Hadamard matrix
    hadamard_mat = hadamard(pixels)           # 1 & -1
    hadamard_plus = (1 + hadamard_mat) / 2    # 1 & 0
    hadamard_minus = (1 - hadamard_mat) / 2   # 0 & 1

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    for i in range(hadamard_mat.shape[0]):
        mask = np.kron(hadamard_plus[i, ...].reshape(resolution), np.ones((factor, factor * 2)))  # 608x1216
        pad_width = int((DMD_resolution[1] * 2 - mask.shape[0]))  # 760
        mask_show = np.pad(mask, ((0, 0), (pad_width - 400, 400)))  # 608x1976
        mask_show = np.rot90(np.rot90(np.rot90(mask_show)))  # 1976x608

        frame = np.uint8(mask_show * 255)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
