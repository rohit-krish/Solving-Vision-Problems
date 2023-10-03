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


def harris_corner_detector(gray, blockSize=3, ksize=3, k=0.05, threshold=0.04, sigma=0):
    """
    gray - Input image, it should be grayscale and float32 type.
    blockSize - It is the size of neighbourhood considered for corner detection
    ksize - Aperture parameter of Sobel derivative used.
    k - Harris detector free parameter in the equation.
    sigma - stddev value for the guassian blur operation
    """

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    Ix2 = cv2.GaussianBlur(Ix**2, (blockSize, blockSize), sigma)
    Iy2 = cv2.GaussianBlur(Iy**2, (blockSize, blockSize), sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (blockSize, blockSize), sigma)

    det_M = (Ix2 * Iy2) - (Ixy**2)
    trace_M = Ix2 + Iy2
    cornerness = det_M - (k * (trace_M**2))

    corners = np.where(cornerness > threshold * cornerness.max())
    return list(zip(corners[1], corners[0]))


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
        for_harris = cv2.GaussianBlur(thresh, (3, 3), 2)
        corners = harris_corner_detector(for_harris, k=0.2)
        for x, y in corners:
            cv2.circle(img, (x, y), 3, (0, 0, 255), cv2.FILLED)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    cv2.imshow("window", stackIt([[img, thresh, for_harris]], [[result, "thresholded"]]))
    if cv2.waitKey() & 0xFF == ord("q"):
        break
