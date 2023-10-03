import cv2
from os import listdir
import numpy as np
from utils import (
    calculate_roundness,
    get_clean_contour,
    get_specific_contour,
    sharpen_image,
    stackIt,
)


final_stack_img = [[], [], []]
final_stack_label = [[], [], []]

for i, img_name in enumerate(listdir("./assets/")):
    img = cv2.imread("./assets/" + img_name)
    h, w, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = sharpen_image(gray_img)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 1)

    _, thresh = cv2.threshold(
        gray_img, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
    )

    contours = get_clean_contour(thresh, w, h)

    result = "?"

    if len(contours) == 1:
        # detect the break which clearly a detached circle
        result = "Breaks"
    else:
        # segmentin the inner circle to check if it has any defects (using the convex hull)
        inner_contour = get_specific_contour(contours, True)
        inner_hull = cv2.convexHull(inner_contour)
        area_diff = np.abs(cv2.contourArea(inner_contour) - cv2.contourArea(inner_hull))

        if area_diff > 200:
            result = "Breaks"
        else:
            # checking notches in the outer contour
            # NOTE: the notches could be present in the inner contour; but since in the given data doesn't contain notches in inner contour, we can assume it won't be happening
            outer_contour = get_specific_contour(contours, False)

            area = cv2.contourArea(outer_contour)
            perimeter = cv2.arcLength(outer_contour, True)
            roundness = calculate_roundness(outer_contour)
            if roundness < 0.89:
                result = "Notches"
            else:
                result = "Good"

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    final_stack_img[i % 3].append(img)
    final_stack_label[i % 3].append(result)


cv2.imshow("window", stackIt(final_stack_img, final_stack_label))
cv2.waitKey()
