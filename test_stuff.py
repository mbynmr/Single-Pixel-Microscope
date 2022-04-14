import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from image_reconstruct import Reconstructor, add_noise, reconstruct_with_other_images_best_masks


def fourier_masks():
    print("reading image")
    image = np.mean(plt.imread("originals/mario64.png"), axis=-1)  # greyscale by mean colour intensity
    plt.imsave("originals/mario64_grey.png", np.real_if_close(image).reshape(image.shape), cmap=plt.get_cmap('gray'))
    r = Reconstructor(image.shape, method='Fourier')
    # m0 = r.matrix[0::4] @ image.flatten()
    # m1 = r.matrix[1::4] @ image.flatten()
    # m2 = r.matrix[2::4] @ image.flatten()
    # m3 = r.matrix[3::4] @ image.flatten()
    # measurements = (m0 - m2) + 1j * (m1 - m3)
    m0 = r.matrix[0::3] @ image.flatten()
    m1 = r.matrix[1::3] @ image.flatten()
    m2 = r.matrix[2::3] @ image.flatten()
    # m0 = m0 + np.multiply(m0 * 0.01, np.random.random(*m0.shape))
    # m1 = m1 + np.multiply(m1 * 0.01, np.random.random(*m1.shape))
    # m2 = m2 + np.multiply(m2 * 0.01, np.random.random(*m2.shape))
    measurements = (m0.dot(2) - m1 - m2) + np.sqrt(3) * 1j * (m1 - m2)
    output = r.reconstruct(measurements)
    print(np.allclose(np.real(output), np.real_if_close(output)))  # todo absolute or real? or imaginary???
    out = np.real(output).reshape(image.shape)
    plt.imshow(out, cmap=plt.get_cmap('gray'))
    plt.imsave("outputs/Fourier.png", out, cmap=plt.get_cmap('gray'))
    print("done")
    plt.show()


def rotating_masks(resolution):
    # rotates masks by 45 degrees
    matrix = Reconstructor(resolution).matrix
    matrix_all = np.zeros([np.asarray(resolution).prod() * 2, np.asarray(resolution).prod()])
    matrix_all[0::2, ...] = (1 + matrix) / 2    # 1 & 0
    matrix_all[1::2, ...] = (1 - matrix) / 2   # 0 & 1
    for i, e in enumerate(matrix_all[100:102, ...]):
        mask = e.reshape(resolution)
        # mask = (np.arange(np.prod(resolution)) / np.prod(resolution)).reshape(resolution)  # todo remov
        # mask = mask + 0.25 * np.diag(np.ones(mask.shape[0]))
        mask_mine_old = my_45_old(mask)
        # for _ in tqdm(range(100)):
        #     mask_mine_old = np.pad(np.rot90(my_45_old(np.kron(mask, np.ones((5, 5)))), axes=(0, 1)), ((144, 144),
        #                                                                                               (22, 22 + 1)))
        mask_mine = my_45(mask)
        # for _ in tqdm(range(100)):
        #     mask_mine = np.pad(np.rot90(my_45(np.kron(mask, np.ones((5, 5)))), axes=(0, 1)), ((144, 144),
        #                                                                                       (22, 22 + 1)))
        big = np.zeros([max(mask.shape[0], mask_mine_old.shape[0], mask_mine.shape[0]),
                        mask.shape[1] + mask_mine_old.shape[1] + mask_mine.shape[1]])
        big[:mask.shape[0], :mask.shape[1]] = mask
        big[:mask_mine_old.shape[0], mask.shape[1]:(mask.shape[1] + mask_mine_old.shape[1])] = mask_mine_old
        big[:mask_mine.shape[0], (mask.shape[1] + mask_mine_old.shape[1]):] = mask_mine
        plt.imshow(big, cmap=plt.get_cmap('gray'))
        plt.show()
    print("done")


def my_45(mask):
    n = mask.shape[0]  # mask is n x n
    m = n - 1  # ((2 * n - 1) - 1) / 2 = m, which is the number of diagonals above (or below) the middle
    my_mask = np.zeros([2 * n - 1, n])  # my_mask is (2n - 1) x n

    my_mask[m, :] = np.diag(mask, k=0)  # middle row

    for d in range(m):  # first rows
        my_mask[d, int(n / 2 - d / 2):int(n / 2 - d / 2 + d + 1)] = np.diag(mask, k=(m - d))

    for d in range(m):  # last rows
        my_mask[2 * m - d, int(n / 2 - d / 2):int(n / 2 - d / 2 + d + 1)] = np.diag(mask, k=(d - m))

    return my_mask
    # 0, 1, 2, 3=diag, 2, 1, 0      - d
    # 3, 2, 1, 0=diag, -1, -2, -3   - np.diag's k = m - d
    # 1, 2, 3, 4=diag, 3, 2, 1      - number of elements = d + 1
    # 0, 0, 1, 2=diag, 1, 0, 0      - floor(d/ 2)
    # 2, 1, 1, 0=diag, 1, 1, 2      - first element = (n / 2) - floor(d/ 2)


def my_45_old(mask):
    n = mask.shape[0]  # mask is n x n
    my_mask = np.zeros([2 * n - 1, n])  # my_mask is (2n - 1) x n

    # ---- first rows ----
    for diagonal in range(n):  # this overwrites row -1, but as long as it is done first, that's not a problem
        for i in range(diagonal):
            my_mask[diagonal - 1, i + 1 + int((n - 1 - diagonal) / 2)] = mask[i, n + i - diagonal]

    # ---- middle row ----
    for i in range(n):  # diagonal = n
        my_mask[n - 1, i] = mask[i, i]

    # ---- last  rows ----
    for d in range(n - 1):  # diagonal = 2 * n - (d + 1)
        for i in range(d + 1):
            my_mask[2 * (n - 1) - d, -i + int((n + d) / 2)] = mask[n - 1 - i, d - i]

    return my_mask


def noise_and_undersampling(resolution):
    file = "mario128.png"

    # matrix = 'Hadamard'
    matrix = 'random'
    r = Reconstructor(resolution, matrix)

    measurements = r.measure(file)
    plt.imsave(f"outputs/output_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))
    noise_type = 'normal'
    # noise_type = 'uniform'
    # measurements = add_noise(measurements, 1e-5, noise_type)
    # plt.imsave(f"outputs/output_noise_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))
    undersample_method = 'last'
    # undersample_method = 'first'
    # undersample_method = 'middle'
    # undersample_method = 'random'
    measurements = r.undersample(measurements, undersample_method, 0.99)
    plt.imsave(f"outputs/output_{0.99}_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))

    # reconstruct_with_other_images_best_masks([32, 32], "Earth4k1.jpg", "Earth4k1.jpg")
