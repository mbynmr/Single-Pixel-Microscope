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


def random_masks(px, frac):
    
    #num_patterns = int(frac*px**2) if int(frac*px**2)%2==0 else int(frac*px**2 + 1)
    
    if (px**2)%2 == 1:
        print("can't do 50/50 masks")
    else:
        num_patterns = int(frac*px**2)
        measurement_matrix = np.zeros((num_patterns, px**2))
        
        row = np.ones(px**2)
        row[:int(px**2/2)] = -1
        
        for i in range(num_patterns):
            np.random.shuffle(row)
            measurement_matrix[i, :] = row
        
    return measurement_matrix
