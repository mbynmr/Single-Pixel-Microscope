from image_reconstruct import Reconstructor

# image resizer: ffmpeg -i mario.png -vf scale=128:-1 mario128.png
# xmodmap -e "keycode 49 = backslash"

# very helpful: https://github.com/cbasedlf/single_pixel_demo/blob/master/sp_demo.py
# will need next: https://github.com/csi-dcsc/Pycrafter6500
# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm
#  always square image dimensions (128x128, 64x64, etc) - is it possible to have non-square dimensions? I think so
#  converts input (test) images to greyscale


def main():
    resolution = [128, 128]  # resolution of reconstruction
    r = Reconstructor(resolution)
    print("measuring...")
    measurements = r.measure()
    print("reconstructing...")
    r.save_image(measurements)
    print("finished!")

    # the code can easily be changed to store the measurements in a file, and reconstruct later. This may be useful in
    # the cases where the reconstructing can be done later but the measurements are time sensitive


if __name__ == '__main__':
    main()
