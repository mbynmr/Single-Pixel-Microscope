import matplotlib.pyplot as plt

from image_reconstruct import Reconstructor, undersample, add_noise, reconstruct_with_other_images_best_masks
# from test_image_deconstruct import test_image, test_image_two
# from output_to_hdmi import output
# from measure_from_multimeter import measure
# from camera import Camera  # todo uncomment
# from notify_run import Notify  # todo uncomment

# image resizer: ffmpeg -i mario.png -vf scale=128:-1 mario128.png
# xmodmap -e "keycode 49 = backslash"

# very helpful: https://github.com/cbasedlf/single_pixel_demo/blob/master/sp_demo.py
# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm opencv-python pyvisa-py notify-run
#  always square image dimensions (128x128, 64x64, etc) - is it possible to have non-square dimensions? I think so
#  converts input (test) images to greyscale
#  use images as masks? may be useful for AI cases "is this a dog?" compressed sensing could do it very fast
#  deconstruct 2 images


def main():
    power = int(5)
    resolution = [2 ** power, 2 ** power]
    print(f"{resolution = }")

    file = "mario128.png"

    r = Reconstructor(resolution)
    measurements = r.measure(file)
    plt.imsave(f"outputs/output_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))
    noise_type = 'normal'
    # noise_type = 'uniform'
    # measurements = add_noise(measurements, 5e-2, noise_type)
    undersample_method = 'last'
    # undersample_method = 'first'
    # undersample_method = 'middle'
    # undersample_method = 'random'
    measurements = undersample(measurements, undersample_method, 0.99)
    plt.imsave(f"outputs/output_{0.99}_{file}", r.reconstruct(measurements), cmap=plt.get_cmap('gray'))

    # reconstruct_with_other_images_best_masks([32, 32], "Earth4k1.jpg", "Earth4k1.jpg")

    # try:
    #     resolution = [16, 16]
    #     c = Camera(resolution)
    #     c.take_picture()
    #     # c.measure()
    #     c.close()
    #     # USB0::0x05E6::0x2100::1269989::INSTR
    #
    # finally:
    #     notify = Notify()
    #     notify.send('Finished')


if __name__ == '__main__':
    main()
