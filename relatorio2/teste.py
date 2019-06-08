import cv2
import numpy as np
from matplotlib import plyplot as plt
img = cv2.imread("impressaodigital.png")
imgCanny = cv2.Canny(img, 150, 240)
# img = cv2.medianBlur(img, 55)
cv2.imshow("original", img)
cv2.imshow("cannyzada", imgCanny)
cv2.waitKey()
