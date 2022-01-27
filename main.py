import cv2 as cv

import line_detection as ld
# import sign_detection as sd

"""
PROGRAM ŁĄCZĄCY WYKRYWANIE ZNAKÓW I LINII NA OBRAZIE
(w czasie rzeczywistym)

[UPDATE 27.01.2022]
robimy tylko jazde pomiedzy liniami; poki co zostawiamy wykrywaniem
znakow, bo jest problem z zainstalowaniem tensorflow'a
"""

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
    print("hello there, I will detect lanes")
    cam = initialize_camera()

    while True:
        # get real time image from camera
        success, frame = cam.read()
        # rotate 180
        frame = cv.rotate(frame, cv.ROTATE_180)
        # display photo
        cv.imshow("Real time footage", frame)
        cv.imshow("Real time footage with lanes", ld.pipeline(frame))



        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()