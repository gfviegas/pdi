import cv2
import numpy as np

img = cv2.imread("MOEDA_VALIOSA.jpg")
img = cv2.medianBlur(img, 5)

lower_green = np.array([120, 180, 100])
upper_green = np.array([140, 200, 120])

#Identificar escrita
edge_canny = cv2.Canny(img, 120, 240)

#K-means
K = 5
Z = img.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
res = res.reshape((img.shape))

cv2.imshow("Apenas tons de cinza", res)
cv2.imshow("Tentando Identificar escrita", edge_canny)
cv2.waitKey()
