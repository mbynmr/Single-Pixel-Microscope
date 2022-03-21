import matplotlib.pyplot as plt

from image_reconstruct import Reconstructor, undersample, add_noise, reconstruct_with_other_images_best_masks


def noise_and_undersampling(resolution):
    file = "mario128.png"

    r = Reconstructor(resolution)
    measurements = r.measure(file)
    plt.imsave(f"outputs/output_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))
    noise_type = 'normal'
    # noise_type = 'uniform'
    measurements = add_noise(measurements, 1e-1, noise_type)
    plt.imsave(f"outputs/output_5e-2_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))
    undersample_method = 'last'
    # undersample_method = 'first'
    # undersample_method = 'middle'
    # undersample_method = 'random'
    # measurements = undersample(measurements, undersample_method, 0.99)
    # plt.imsave(f"outputs/output_{0.99}_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))

    # reconstruct_with_other_images_best_masks([32, 32], "Earth4k1.jpg", "Earth4k1.jpg")
