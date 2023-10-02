from mylib.show import stackIt
import cv2
from os import listdir
import numpy as np

colors = [(252, 3, 3), (252, 186, 3), (78, 252, 3), (3, 44, 252), (252, 3, 231)]  # RGB

for img_name in listdir("./assets/"):
    img = cv2.imread("./assets/" + img_name)
    # h, w, _ = img.shape
    # img = img[(h // 2) + 5 : -10, ...]
    img = cv2.GaussianBlur(img, (3, 3), 2)
    edges = cv2.Canny(img, 50, 100)
    edges = cv2.dilate(edges, (3, 3), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # finding the 5 Glue bottle contours
    gb_contour_indxes = [[i, cv2.contourArea(cnt)] for i, cnt in enumerate(contours)]
    gb_contour_indxes = sorted(gb_contour_indxes, key=lambda x: x[1], reverse=True)[:5]
    gb_contour_indxes = [*map(lambda x: x[0], gb_contour_indxes)]
    gb_contours = [contours[idx] for idx in gb_contour_indxes]

    gb_mask = np.zeros_like(img, dtype="uint8")

    for i, gb_cnt in enumerate(gb_contours):
        cv2.drawContours(gb_mask, [gb_cnt], -1, colors[i], cv2.FILLED)

    cv2.drawContours(img, gb_contours, -1, (0, 255, 0), 1)

    cv2.imshow("window", stackIt([[img, edges], [gb_mask]]))
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
