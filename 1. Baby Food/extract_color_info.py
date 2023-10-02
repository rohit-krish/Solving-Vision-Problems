from mylib.show import stackIt
import numpy as np
import cv2
from os import listdir


def onChange(*_):
    pass


cv2.namedWindow("TrackBars")

cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, onChange)
cv2.createTrackbar("Sat Min", "TrackBars", 161, 255, onChange)
cv2.createTrackbar("Val Min", "TrackBars", 142, 255, onChange)

cv2.createTrackbar("Hue Max", "TrackBars", 183, 255, onChange)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, onChange)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, onChange)

for img_path in listdir("./assets/"):
    img = cv2.imread("./assets/" + img_path)

    while 1:
        if cv2.waitKey(1) == ord("q"):
            break

        # to HSV space
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, s_min, v_min, h_max, s_max, v_max)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)  # input img,output img

        cv2.imshow(
            "TrackBars",
            stackIt(
                [[img, imgHSV], [mask, imgResult]],
                [["img", "hsv"], ["mask", "result"]],
                img_scale=0.7,
            ),
        )
