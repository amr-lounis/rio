import sys
import time
import random
import logging
import threading, queue
 
def Transmitter():
    """Transmits readings to PC at 1 second intervals on a separate thread"""
    logging.debug(f'[Transmitter] Starting')

    # Start independent loop sending values
    i = 0
    while True:
       reading = random.randint(0,100)
       logging.debug(f'[Transmitter] Iteration: {i}, reading: {reading}')
       i += 1
       time.sleep(1)

if __name__ == '__main__':

    # Set up logging - very advisable with threaded code
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')

    # Create independent thread to transmit readings
    thr = threading.Thread(target=Transmitter, args=())
    thr.start()

    # Main loop - waiting for commands from PC
    logging.debug('[Main] Starting main loop')
    i = 0
    while True:
       # Not what you want but just showing the other thread is unaffacted by sleeping here
       time.sleep(5)

       i += 1
       logging.debug(f'[Main] Iteration: {i}')