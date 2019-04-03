# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot(img, cdf, title):
    plt.plot(cdf, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.legend(('cdf', title), loc='upper left')
    plt.show()


img = cv2.imread('./menina.jpg')

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
plot(img, cdf, "before")

cdf_normalized = cdf * hist.max() / cdf.max()
plot(img, cdf_normalized, "after")
