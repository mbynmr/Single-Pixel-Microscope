from notify_run import Notify

# from noise_and_undersampling import noise_and_undersampling
from camera import Camera

# image resizer: ffmpeg -i mario.png -vf scale=128:-1 mario128.png
# xmodmap -e "keycode 49 = backslash"

# very helpful: https://github.com/cbasedlf/single_pixel_demo/blob/master/sp_demo.py
# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm opencv-python pyvisa-py notify-run
#  always square image dimensions (128x128, 64x64, etc) - is it possible to have non-square dimensions? I think so
#  converts input (test) images to greyscale
#  use images as masks? may be useful for AI cases "is this a dog?" compressed sensing could do it very fast
#  deconstruct 2 images and compare the measurements to tell how alike images are? Can't see how this is useful
#  our camera sets minimum pixel values to min 0 & max 255, so there isn't a real-world transmission measurement
#  non-binary masks? would that help? probably not?


def main():
    power = int(5)  # 4: 16x16, 5: 32x32, 6: 64x64, 7: 128x128, 8: 256x256

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
