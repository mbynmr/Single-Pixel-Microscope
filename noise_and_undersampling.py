import matplotlib.pyplot as plt

from image_reconstruct import Reconstructor, add_noise, reconstruct_with_other_images_best_masks


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
