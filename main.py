import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

video = cv2.VideoCapture("road_car_view.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (width, height))
x = time.time()

while time.time() - x < 85:
    ret, or_frame = video.read()
    frame = cv2.GaussianBlur(or_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_y = np.array([20, 90, 125])
    upper_y = np.array([50, 255, 255])

    mask = cv2.inRange(hsv, lower_y, upper_y)
    # cv2.imshow("mask", mask)
    edges = cv2.Canny(mask, 74, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)
    out.write(frame)
    key = cv2.waitKey(1)
    if (key == 27):
        break

video.release()
cv2.destroyAllWindows()
