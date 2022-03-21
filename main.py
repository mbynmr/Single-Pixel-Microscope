import matplotlib.pyplot as plt

from noise_and_undersampling import noise_and_undersampling
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
#  deconstruct 2 images and compare the measurements to tell how alike images are? Can't see how this is useful


def main():
    power = int(6)
    # 4: 16x16, 5: 32x32, 6: 64x64, 7: 128x128, 8: 256x256
    resolution = [2 ** power, 2 ** power]
    print(f"{resolution = }")

    noise_and_undersampling(resolution)

    # try:
    #     c = Camera(resolution)
    #     c.take_picture()
    #     # c.measure()
    #     c.close()
    # finally:
    #     notify = Notify()
    #     notify.send('Finished')


if __name__ == '__main__':
    main()
