# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:43:05 2019

@author: Hristo Dinkov, Roboclub TU-Sofia
"""

import cv2
from time import time, sleep
from sys import argv, exit, stdout
from os import mkdir, path
import errno


def main():
    if len(argv) != 3:
        print("""Usage: %s DIRECTORY SECONDS
Record frames from the camera for SECONDS seconds and save them in DIRECTORY""" % argv[0])
        exit(1)
    name = argv[1]
    seconds = int(argv[2])
    record(name, seconds)


def status(text):
    """Helper function to show status updates"""
    stdout.write('\r%s' % text)
    stdout.flush()


def record(dirname, seconds):
    """ Record from the camera """

    delay = 3 # Give people a 3 second warning to get ready
    started = time()
    while time() - started < delay:
        status("Recording in %.0f..." % max(0, delay - (time() - started)))
        sleep(0.1)

    try:
        mkdir(dirname)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise err

    num_frames = 0
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
    
    
    started = time()
    while time() - started < seconds or seconds == -1:
        ret, frame = cap.read()
        
        cv2.imwrite(path.join(dirname, '%05d.png') % num_frames, frame)
        
        num_frames += 1

        # Update our progress
        if seconds != -1:
            status("Recording [ %d frames, %3.0fs left ]" % (num_frames,
                                                             max(0, seconds - (time() - started))))
        else:
            status("Recording [ %d frames, %3.0fs so far ]" % (num_frames,
                                                               max(0, (time() - started))))
    
    cap.release()
    cv2.destroyAllWindows()
    
    print('')

    # Save the frames to a file, appending if one already exists
    print('Wrote %d frames to %s\n' % (num_frames, dirname))


if __name__ == '__main__':
    main()