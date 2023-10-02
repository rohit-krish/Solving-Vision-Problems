from mylib.show import stackIt
import cv2
from os import listdir
import numpy as np


# colors = [(252, 3, 3), (252, 186, 3), (78, 252, 3), (3, 44, 252), (252, 3, 231)]  # RGB
def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


for img_name in listdir("./assets/"):
    img = cv2.imread("./assets/" + img_name)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    edges = cv2.Canny(img, 50, 100)
    edges = cv2.dilate(edges, (3, 3), iterations=1)

    outer_contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # finding the 5 Glue bottle contours
    outer_contour_indxes = [
        [i, cv2.contourArea(cnt)] for i, cnt in enumerate(outer_contours)
    ]
    outer_contour_indxes = sorted(
        outer_contour_indxes, key=lambda x: x[1], reverse=True
    )[:5]
    outer_contour_indxes = [*map(lambda x: x[0], outer_contour_indxes)]
    outer_contours = [outer_contours[idx] for idx in outer_contour_indxes]

    cv2.imshow("window", stackIt([[img, edges]]))

    for i, gb_cnt in enumerate(outer_contours):
        mask = np.zeros_like(edges, dtype="uint8")
        cv2.drawContours(mask, [gb_cnt], -1, 255, cv2.FILLED)
        bottle = cv2.bitwise_and(img, img, mask=mask)
        h, w, _ = bottle.shape
        bottle = bottle[(h // 2) + 20 : -15, ...]
        bottle = sharpen_image(bottle)
        bottle = cv2.GaussianBlur(bottle, (3, 3), 1)

        bottle_edges = cv2.Canny(bottle, 50, 200)
        bottle_edges = cv2.dilate(bottle_edges, (5, 5), iterations=1)

        inner_contours, _ = cv2.findContours(
            bottle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        inner_contours = [cv2.approxPolyDP(cnt, 3, False) for cnt in inner_contours]

        len_inner_contours = len(inner_contours)

        print(len(inner_contours[0]))

        if len_inner_contours == 1:
            result = "No Label"
        elif len_inner_contours > 2:
            result = "Torn Label"
        elif len_inner_contours == 2:
            inner_contours = [inner_contours[np.argmax([cv2.contourArea(cnt) for cnt in inner_contours])]]

            # decide if it is straight or tilted (solution. checking the difference in minAreaRect and normal rect)

            min_area_rect = list(cv2.minAreaRect(inner_contours[0]))
            box = cv2.boxPoints(min_area_rect).astype(int)
            
            min_x = int(np.min(box[:, 0]))
            max_x = int(np.max(box[:, 0]))
            min_y = int(np.min(box[:, 1]))
            max_y = int(np.max(box[:, 1]))

            min_rect_w, min_rect_h = max_x - min_x, max_y - min_y

            x, y, w, h = cv2.boundingRect(inner_contours[0])

            area_diff = abs((w*h) - (min_rect_w * min_rect_h))
            if area_diff < 400:
                result = 'Good Label'
            else:
                result = 'deciding'
                inner_contours = [cv2.approxPolyDP(inner_contours[0], 3, True)]
                if len(inner_contours[0]) == 4:
                    result = 'Good Label'
                else:
                    result = 'Torn Label'

        cv2.drawContours(bottle, inner_contours, -1, 255, 1)
        cv2.imshow("bottle", stackIt([[bottle]], [[result]]))
        cv2.waitKey()

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
