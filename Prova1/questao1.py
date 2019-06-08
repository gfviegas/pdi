import cv2
import numpy as np


img = cv2.imread("wally.jpg")

lower_white_BGR = np.array([200, 200, 200])
upper_white_BGR = np.array([255, 255, 255])

lower_red_BGR = np.array([0, 0, 140])
upper_red_BGR = np.array([80, 80, 255])

REDmaskBGR = cv2.inRange(img, lower_red_BGR, upper_red_BGR)
WHITEmaskBGR = cv2.inRange(img, lower_white_BGR, upper_white_BGR)

resWHITE = cv2.bitwise_and(img, img, mask=WHITEmaskBGR)
resRED = cv2.bitwise_and(img, img, mask=REDmaskBGR)


resFinal = resWHITE + resRED
cv2.imshow("resultado", resFinal)
cv2.waitKey();
