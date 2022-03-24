import usb.core
import usb.util

# from infi.devicemanager import DeviceManager
# dm = DeviceManager()
# devices = dm.all_devices
# for i in devices:
#     try:
#         print ('{} : address: {}, bus: {}, location: {}'.format(i.friendly_name, i.address, i.bus_number, i.location))
#     except Exception:
#         pass


import usb.backend.libusb1


def enumerate_usb():    #I use a simple function that scans all known USB connections and saves their info in the file
    with open("EnumerateUSBLog.txt", "w") as wf:
        counter = 0
        for d in dev:
            try:
                wf.write("USB Device number " + str(counter) + ":" + "\n")
                wf.write(d._get_full_descriptor_str() + "\n")
                wf.write(d.get_active_configuration() + "\n")
                wf.write("\n")
                counter += 1
            except NotImplementedError:
                wf.write("Device number " + str(counter) + "is busy." + "\n")
                wf.write("\n")
                counter += 1
            except usb.core.USBError:
                wf.write("Device number " + str(counter) + " is either disconnected or not found." + "\n")
                wf.write("\n")
                counter += 1
        wf.close()

backend = usb.backend.libusb1.get_backend(
    find_library=lambda x: "C:\Anaconda3\envs\Single-Pixel-Microscope\VS2019\MS64\Release\dll\libusb-1.0.dll"
)
dev = usb.core.find(backend=backend, find_all=True)
enumerate_usb()
