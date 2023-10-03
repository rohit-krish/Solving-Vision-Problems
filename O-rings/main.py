import cv2
from os import listdir

for img_name in listdir("./assets/"):
    img = cv2.imread("./assets/" + img_name)

    cv2.imshow("window", img)
    if cv2.waitKey() & 0xFF == ord("q"):
        break
