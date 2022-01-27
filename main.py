"""
PROGRAM ŁĄCZĄCY WYKRYWANIE ZNAKÓW I LINII NA OBRAZIE
(w czasie rzeczywistym)

[UPDATE 27.01.2022]
robimy tylko jazde pomiedzy liniami; poki co zostawiamy wykrywaniem
znakow, bo jest problem z zainstalowaniem tensorflow'a

Usage:
    manage.py [--debug] [--show_camera]

Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    -f --file=<file>   A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value> Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
"""
import cv2 as cv
from docopt import docopt
from matplotlib.pyplot import show

import line_detection as ld
# import sign_detection as sd


def initialize_camera():
    # camera resolution
    frameWidth = 640
    frameHeight = 480
    camera = cv.VideoCapture(0)

    # initialize the camera
    camera.set(3, frameWidth)
    camera.set(4, frameHeight)
    return camera


def main():
    arguments = docopt(__doc__)
    debug = arguments["--debug"]
    show_camera = arguments["--show_camera"]
    # print(arguments)

    if(debug):
        print("hello there, I will detect lanes")
    cam = initialize_camera()

    while True:
        # get real time image from camera
        success, frame = cam.read()
        # rotate 180
        frame = cv.rotate(frame, cv.ROTATE_180)
        # get a copy with lines on
        frame_with_lines = frame.copy()
        frame_with_lines = ld.pipeline(frame_with_lines)
        # display photo
        if(show_camera):
            cv.imshow("Real time footage", frame)
            cv.imshow("Real time footage with lanes", frame_with_lines)



        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()