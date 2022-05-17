import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from image_reconstruct import Reconstructor, add_noise, reconstruct_with_other_images_best_masks
from HadamardOrdering import Walsh


def fourier_masks():
    print("reading image")
    image = np.mean(plt.imread("originals/mario64.png"), axis=-1)  # greyscale by mean colour intensity
    plt.imsave("originals/mario64_grey.png", np.kron(image, np.ones((5, 5))), cmap=plt.get_cmap('gray'))
    # mask_method = 'Fourier'
    mask_method = 'Fourier_2D'
    r = Reconstructor(image.shape, method=mask_method)
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

    # undersample cutoff method
    undersample_method = 'Sharp'
    # undersample_method = 'Scale'
    # undersample_method = 'Dither'

    # orderings
    walsh = False
    # walsh = True
    # cc = False
    cc = True
    if walsh or cc:
        _, order_walsh = Walsh(image.shape[0], np.zeros([np.prod(image.shape), np.prod(image.shape)]), True)
        _ = None  # remove from memory
        order_walsh = np.int_(order_walsh)
        # walsh = hadamard[order_walsh]
        # hadamard[order_walsh] = walsh
        if cc:
            order_cc = cake_cutting(image.shape)
            # cake_cut = hadamard[order]
            # hadamard[order] = cake_cut
        else:
            order_cc = range(len(measurements))
    else:
        order_walsh = range(len(measurements))
        order_cc = range(len(measurements))

    # frac = [1 / 2 ** 0, 1 / 2 ** 1, 1 / 2 ** 2, 1 / 2 ** 3]  # 1, 1/2, 1/4, 1/8
    frac = 1 / 2 ** np.arange(4)
    for f in frac:
        meas = np.array(measurements)
        # order change
        meas[order_walsh] = np.array(measurements)  # from "Walsh" to "Natural" Hadamard analogue
        meas[order_cc] = np.array(meas)  # inverse of "Natural" to cc
        # meas = meas[order_cc]  # from "Natural" to "Cake Cutting" Hadamard analogue
        # ---- undersample ----
        if f < 1:
            meas = undersample(meas, f, undersample_method)
        # undo order change
        meas = meas[order_cc]
        # meas[order_cc] = np.array(meas)
        meas = meas[order_walsh]
        print(f"sampling: {100 * np.sum(np.where(meas != 0, 1, 0)) / len(meas)}%")
        output = r.reconstruct(meas)
        out = np.real(output).reshape(image.shape)
        # plt.imshow(out, cmap=plt.get_cmap('gray'))
        # plt.show()
        plt.imsave(f"outputs/Fourier_{f}.png", np.kron(out, np.ones((5, 5))), cmap=plt.get_cmap('gray'))
    print("done")


def undersample(meas, f, method=None):
    if method is None or method == 'Sharp':
        meas[int(len(meas) * f):] = 0
        return meas
    if method == 'Scale':
        meas[int(len(meas) * f):] = 0
        length = 10
        meas[int(len(meas) * f - length):int(len(meas) * f)] = (
                ((np.arange(length)[::-1] + 1) / length) * meas[int(len(meas) * f - length):int(len(meas) * f)])
        return meas
    if method == 'Dither':
        # length = int(len(meas) * f / 10)
        length = 50
        out = np.zeros([length, 1, 3])  # 1D RGB image
        for colour in range(3):
            out[:, 0, colour] = np.arange(length)[::-1] / (length - 1)
        plt.imsave("outputs/temp_1.png", out)
        Image.open("outputs/temp_1.png").convert("1").save("outputs/temp_2.png")  # load, dither, save
        image = plt.imread("outputs/temp_2.png", out)
        dither = image[np.amin(np.nonzero(1 - image[:, 0])):(np.amax(np.nonzero(image[:, 0])) + 1), 0]
        print(len(dither))
        # exclude unnecessary zeros at the end of, and ones at the start of, the dither
        multiplier = np.ones(meas.shape)
        # (cutoff - num of ones) to (cutoff + num of zeros) is the new dither zone
        multiplier[int(len(meas) * f - len(np.nonzero(dither)[0])):
                   int(len(meas) * f + len(dither)) - len(np.nonzero(dither)[0])] = dither
        multiplier[int(len(meas) * f + len(dither) - len(np.nonzero(dither)[0])):] = 0
        return meas * multiplier
    else:
        raise NotImplementedError(f"no '{method}' method")


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


def cake_cutting(resolution):  # returns cake cutting ordering array, requiring Hadamard to be used
    pixels = np.asarray(resolution).prod()
    from scipy.linalg import hadamard
    hadamard_matrix = hadamard(pixels)
    hadamard_matrix[hadamard_matrix == -1] = 0
    totals = np.zeros(pixels)
    from skimage import measure
    for i, mask in enumerate(hadamard_matrix):
        mask = mask.reshape(resolution)
        totals[i] = measure.label(mask, return_num=True, connectivity=1)[1] + measure.label(1 - mask, return_num=True,
                                                                                            connectivity=1)[1]
    return sorted(range(len(totals)), key=lambda k: totals[k])  # todo sort from stackoverflow.com/questions/7851077
