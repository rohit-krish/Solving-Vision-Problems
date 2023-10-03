import cv2
from os import listdir
from mylib.show import stackIt
import numpy as np
from segment import segment_objects, colorize_labels


def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def get_clean_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) / (w * h)) > 0.02]

    thresh = np.zeros_like(thresh, dtype="uint8")
    cv2.drawContours(thresh, contours, -1, 255, cv2.FILLED)
    return thresh, contours


for img_name in listdir("./assets/"):
    img = cv2.imread("./assets/" + img_name)
    h, w, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = sharpen_image(gray_img)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 1)

    _, thresh = cv2.threshold(
        gray_img, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
    )

    thresh, contours = get_clean_contour(thresh)

    result = "?"
    if len(contours) == 1:
        result = "Breaks"
    else:
        # segmentin the inner circle to check if it has any defects
        inv_thresh = cv2.bitwise_not(thresh)
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        inner_contour = contours[np.argmin(contour_areas)].squeeze()
        
        # center points of the circle
        cx = inner_contour[:, 0].mean().astype('int32')
        cy = inner_contour[:, 1].mean().astype('int32')

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    # segmented = segment_objects(thresh).astype("uint8")
    # segmented = colorize_labels(segmented)

    cv2.imshow("window", stackIt([[img, thresh]], [[result, "thresholded"]]))
    if cv2.waitKey() & 0xFF == ord("q"):
        break
