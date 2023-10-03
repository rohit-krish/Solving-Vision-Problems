import numpy as np
import cv2


def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def get_clean_contour(thresh, w, h):
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) / (w * h)) > 0.02]
    return contours


def get_specific_contour(contours, is_inner):
    """
    returns inner or outer contours
    """
    np_func = np.argmin if is_inner else np.argmax
    contour_areas = [cv2.contourArea(cnt) for cnt in contours]
    inner_contour = contours[np_func(contour_areas)].squeeze()
    return inner_contour


def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    roundness = (4 * np.pi * area) / (perimeter**2)
    return roundness  # for a perfect circle it will yield 1


def stackIt(img_matrix, label_matrix=None, img_scale=1, label_height=30, **kwargs):
    """
    just for visualizing purpose, not nessesory.
    """
    if len(kwargs) == 0:
        kwargs = {
            "fontFace": 0,
            "fontScale": 0.8,
            "color": (255, 255, 255),
            "thickness": 1,
        }

    label_height = 0 if label_matrix == None else label_height
    img_matrix = _resize_and_fill_gaps(img_matrix, img_scale, label_height)

    # putting the labels in each images
    if label_matrix:
        for img_row, label_row in zip(img_matrix, label_matrix):
            for image, label in zip(img_row, label_row):
                h, *_ = image.shape
                cv2.putText(image, label, (10, h - 10), **kwargs)

    row_images = [np.hstack(row) for row in img_matrix]
    return np.vstack(row_images)


def _resize_and_fill_gaps(matrix, scale, label_height):
    # height and width of the first image
    height, width, *_ = matrix[0][0].shape
    height = int(height * scale)
    width = int(width * scale)

    n_rows = len(matrix)
    n_cols = max(len(row) for row in matrix)

    result = np.zeros((n_rows, n_cols, height + label_height, width, 3), dtype=np.uint8)

    for r_idx, row in enumerate(matrix):
        for c_idx, img in enumerate(row):
            img = np.squeeze(img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img, (width, height))
            text_place = np.zeros((label_height, width, 3), dtype=np.uint8)
            result[r_idx, c_idx] = np.vstack((img, text_place))

    return result

