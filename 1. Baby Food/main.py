from mylib.show import stackIt
import cv2
from os import listdir
import numpy as np
from itertools import combinations
from sklearn.cluster import DBSCAN


def detect_spoon_roi(img):
    # [0, 124, 137], [179, 255, 255] :: HSV [min, max] for precise color match
    # [0, 133, 64],  [188, 255, 255] :: HSV [min, max] for loose color match (have more space around)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHSV[..., 2] = cv2.equalizeHist(imgHSV[..., 2])

    lower = np.array([0, 161, 142])
    upper = np.array([183, 255, 255])

    mask = cv2.inRange(imgHSV, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.bitwise_and(img, img, mask=mask)


def non_max_suppression(lines, thresh):
    if (lines is None) or (len(lines) == 0):
        return []

    lines = lines.squeeze().tolist()

    for count, current_line in enumerate(lines):
        current_rho, current_theta = current_line

        for line in lines[count + 1 :]:
            rho, theta = line

            delta_rho = abs(rho - current_rho)

            if delta_rho < thresh:
                lines.remove(line)

    return lines


def hough_line_detection(edges, img1, img2):
    def draw(img, lines):
        if lines is not None:
            for line in lines:
                rho, theta = line.squeeze()
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    draw(img1, lines)

    nms_lines = np.array(non_max_suppression(lines, 10))
    # print(nms_lines.shape)
    draw(img2, nms_lines)

    return nms_lines


def cluster_lines(lines):
    distances = []
    angles = []
    for line1, line2 in combinations(lines, 2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        # Calculate the Euclidean distance between two lines in the (rho, theta) space
        distance = np.sqrt((rho1 - rho2) ** 2 + (theta1 - theta2) ** 2)

        # Append the distance and angle to the respective lists
        distances.append(distance)
        angles.append(np.abs(theta1 - theta2))

    distances = np.array(distances)
    angles = np.array(angles)

    X = np.column_stack((distances, angles))

    db = DBSCAN(eps=2000, min_samples=1, metric="euclidean").fit(X)
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(num_clusters)


for img_path in listdir("./assets/"):
# for img_path in ["BabyFood-Sample1.JPG"]:
    img = cv2.imread("./assets/" + img_path)
    img_contours = img.copy()
    img_blank_approx_cnt = np.zeros(img.shape[:2], dtype="uint8")
    img_with_lines = img.copy()

    spoon_roi = detect_spoon_roi(img)

    edges = cv2.Canny(spoon_roi, 150, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, hieararchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        threshold_area = np.percentile(contour_areas, 90)
        if threshold_area > 15_000:
            text = "more than one"
        elif threshold_area < 500:
            text = "zero"
        elif threshold_area < 4000:
            text = "one"
        else:
            text = "?"

        filtered_contours = [
            (cnt, area)
            for cnt, area in zip(contours, contour_areas)
            if area >= threshold_area
        ]


        for cnt, area in filtered_contours:
            if area > 1000:
                cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), 1)
                approx_contours = cv2.approxPolyDP(cnt, 5, False)
                cv2.drawContours(img_blank_approx_cnt, [approx_contours], -1, 255, 1)

        img_blank_approx_cnt = cv2.dilate(img_blank_approx_cnt, kernel, iterations=1)
        lines = hough_line_detection(img_blank_approx_cnt, img, img_with_lines)

        if text == "?":
            if len(lines) == 1:
                text = "one?"
            else:
                cluster_lines(lines)
    else:
        text = "zero"

    cv2.imshow(
        "window",
        stackIt(
            [
                [img, spoon_roi, edges],
                [img_contours, img_blank_approx_cnt, img_with_lines],
            ],
            [
                ["all lines", "spoon roi", "edges"],
                ["contours", "approx contours", "nms lines - " + text],
            ],
            0.7,
        ),
    )
    if cv2.waitKey() == ord("q"):
        break


cv2.destroyAllWindows()
