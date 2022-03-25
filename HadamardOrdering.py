# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:00:53 2022

@author: ppyet1
"""
import numpy as np
from scipy.linalg import hadamard


def Walsh(px, hadamard_matrix):
    N = px ** 2
    # gray code 
    m = np.log2(N)

    walsh_matrix = np.zeros((N, N))
    n_w_array = np.zeros(N)

    # reorder
    for n_h in range(N):
        g = np.binary_repr(n_h)
        # leading zeros
        g = int(m - len(g)) * '0' + g

        # index in walsh ordered matrix
        n_w = g[-1::-1]
        # binary to decimal
        n_w = np.sum([int(g[int(i)]) * 2 ** i for i in np.arange(m)])

        # change rows
        walsh_matrix[int(n_w), :] = hadamard_matrix[n_h, :]

        # n_w_array[n_h]= n_w

    return walsh_matrix
