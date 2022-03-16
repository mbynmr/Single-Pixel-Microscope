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
    try:
        resolution = [16, 16]
        c = Camera(resolution)
        c.take_picture()
        # c.measure()
        c.close()
        # USB0::0x05E6::0x2100::1269989::INSTR

        # reconstruct_with_other_images_best_masks([32, 32], "Earth4k1.jpg", "Earth4k1.jpg")
    finally:
        notify = Notify()
        notify.send('Finished')


if __name__ == '__main__':
    main()
