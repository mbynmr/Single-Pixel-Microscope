# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:00:53 2022

@author: ppyet1
"""
import numpy as np
from scipy.linalg import hadamard
import graycode
from tqdm import tqdm


def Walsh(px, hadamard_matrix, save_order=False):
    N = px ** 2
    # gray code 
    m = int(np.log2(N))

    walsh_matrix = np.zeros((N, N))
    n_w_array = np.zeros(N)

    # arr = graycode.gen_gray_codes(m)
    # print(arr)

    # reorder
    for n_h in range(N):
        g = np.binary_repr(graycode.tc_to_gray_code(n_h))
        # leading zeros
        g = int(m - len(g)) * '0' + g

        # index in walsh ordered matrix
        n_w = g[::-1]
        # binary to decimal
        n_w = np.sum([int(g[int(i)]) * 2 ** i for i in np.arange(m)])

        n_w_array[n_h] = n_w

        # change rows
        walsh_matrix[n_h, :] = hadamard_matrix[[int(n_w)], :]

    if save_order:
        return walsh_matrix, n_w_array
    else:
        return walsh_matrix


def random_masks(pixels, frac, seed):
    # num_patterns = int(frac*px**2) if int(frac*px**2)%2==0 else int(frac*px**2 + 1)

    if pixels % 2 == 1:
        raise ValueError(f"Can't do 50/50 masks for odd numbered mask shapes: {pixels = }")
    num_patterns = int(frac * pixels)
    measurement_matrix = np.zeros((num_patterns, pixels))

    row = np.ones(pixels)
    row[:int(pixels / 2)] = -1

    rs = np.random.RandomState(seed)
    seeds = rs.randint(low=int(0), high=int(num_patterns * 1000), size=num_patterns)

    for i in tqdm(range(num_patterns)):
        np.random.RandomState(seeds[i]).shuffle(row)
        measurement_matrix[i, :] = row

    return measurement_matrix
    # rs = np.random.RandomState(seed=np.random.MT19937(np.random.SeedSequence(123456789)))
