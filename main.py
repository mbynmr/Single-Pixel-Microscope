from image_reconstruct import Reconstructor
from image_reconstruct import reconstruct_with_other_images_best_masks
# from test_image_deconstruct import test_image, test_image_two
# from output_to_hdmi import output
# from measure_from_multimeter import measure
from camera import Camera
from notify_run import Notify

# image resizer: ffmpeg -i mario.png -vf scale=128:-1 mario128.png
# xmodmap -e "keycode 49 = backslash"

# very helpful: https://github.com/cbasedlf/single_pixel_demo/blob/master/sp_demo.py
# will need next: https://github.com/csi-dcsc/Pycrafter6500
# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm pyusb opencv-python
#  always square image dimensions (128x128, 64x64, etc) - is it possible to have non-square dimensions? I think so
#  converts input (test) images to greyscale


def main():
    power = int(4)  # 4: 16x16, 5: 32x32, 6: 64x64, 7: 128x128, 8: 256x256

    xplc_index = int(2)  # 0 to 3: [0.02, 0.1, 1, 10]
    measurements_per_mask = int(3)

    fraction = 1
    # method = 'Hadamard'
    method = 'Walsh'
    # method = 'Random'

    try:
        resolution = [2 ** power, 2 ** power]
        print(f"{resolution = }")
        xplc = [0.02, 0.1, 1, 10]  # these are the only options for the multimeter

        c = Camera(resolution, xplc[xplc_index], measurements_per_mask, fraction, method)
        c.take_picture(5)  # input pause time in seconds before the masks are shown
        c.close()
    finally:
        notify = Notify()
        notify.send('Finished')
        # Notify().send('Finished')


if __name__ == '__main__':
    main()
