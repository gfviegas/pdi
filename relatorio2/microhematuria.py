import numpy as np
import cv2

# Read image
img = cv2.imread("MicroHematuria.jpg", cv2.IMREAD_GRAYSCALE)

edge_canny = cv2.Canny(img, 150, 240)

Z = edge_canny.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 1.0)
K = 8
ret, label, center = cv2.kmeans(
    Z, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((edge_canny.shape))

cv2.imwrite("binarizada.jpg", res2)
