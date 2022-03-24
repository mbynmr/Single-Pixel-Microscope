import numpy as np
from tqdm import tqdm
import pyvisa as visa


def measure():
    # Open an instance of the Resource manager class and assign the handle rm
    rm = visa.ResourceManager()
    # Show available resources
    print(rm.list_resources())
    # Open an instance of the USB resource class and assign the object handle my_instrument
    my_instrument = rm.open_resource('USB0::0x05E6::0x2100::1269989::INSTR')
    my_instrument.timeout = None  # sets no instrument timeout

    # Configure the instrument to measure DC Voltage, with a maximum range, and the minimum resolution
    my_instrument.write('CONFigure:VOLTage:DC 10, MIN', )
    # Set the trigger source to internal, and immediate, set the trigger delay to auto, and set the instrument to take 10
    # samples after the trigger event
    my_instrument.write('TRIGger:SOURce IMMediate')
    my_instrument.write('TRIGger:DELay:AUTO ON')
    my_instrument.write('SAMPle:COUNt 1')

    my_instrument.write('TRIG:COUN INF')  # infinite triggers

    # Set the instrument to a wait for trigger state. In the case of the Immediate internal trigger, the instrument will
    # send a trigger as soon as instrument is set to this state
    my_instrument.write('INITiate')
    # Fetch the data from the instruments internal storage to the buffer, and read in the data.
    for _ in tqdm(range(1000)):
        y = my_instrument.query('FETCh?')

    # clear and close the instance of the instrument, and close the resource manager
    my_instrument.clear()
    my_instrument.close()
    rm.close()
