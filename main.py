from Pycrafter6500_master import pycrafter6500

from image_reconstruct import Reconstructor
from test_image_deconstruct import test_image, test_image_two
from output_to_hdmi import output

# image resizer: ffmpeg -i mario.png -vf scale=128:-1 mario128.png
# xmodmap -e "keycode 49 = backslash"

# very helpful: https://github.com/cbasedlf/single_pixel_demo/blob/master/sp_demo.py
# will need next: https://github.com/csi-dcsc/Pycrafter6500
# todo
#  python 3.8 environment: pip install numpy matplotlib scikit-image tqdm pyusb pycrafter4500
#  always square image dimensions (128x128, 64x64, etc) - is it possible to have non-square dimensions? I think so
#  converts input (test) images to greyscale


def main():
    resolution = [32, 32]  # resolution of reconstruction
    r = Reconstructor(resolution)
    print("measuring...")
    measurements = r.measure()
    print("reconstructing...")
    r.save_image(measurements)
    print("finished!")

    # the code can easily be changed to store the measurements in a file, and reconstruct later. This may be useful in
    # the cases where the reconstructing can be done later but the measurements are time sensitive


def test():
    import usb.core
    import usb.util  # todo remove
    import usb.backend.libusb1

    backend = usb.backend.libusb1.get_backend(
        find_library=lambda x: "C:\Anaconda3\envs\Single-Pixel-Microscope\VS2019\MS64\Release\dll\libusb-1.0.dll"
    )

    # devs = usb.core.find(backend=backend, find_all=True)
    # for dev in devs:
    #     if dev is None:
    #         raise ValueError('Device not found')
    #
    #     # cfg = dev.get_active_configuration()
    #     try:
    #         cfg = dev.get_active_configuration()
    #     except NotImplementedError:
    #         print("can't get_active_configuration() for this device")
    #     else:
    #         print("get_active_configuration() for this device:")
    #         print(cfg)

    controller = pycrafter6500.dmd(backend)
    controller.idle_on()
    # sets the DMD to idle mode

    controller.idle_off()
    # wakes the DMD from idle mode

    controller.standby()
    # sets the DMD to standby

    print("done")


def test_4500():
    import pycrafter4500
    pycrafter4500.power_up()


def test_fullscreen():
    output()


if __name__ == '__main__':
    # main()
    # test()
    test_fullscreen()
    # test_4500()
    # test_image()
    # test_image_two()
