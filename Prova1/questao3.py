import cv2
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

#exibe o histograma
def plot(img, cdf, title):
    plt.plot(cdf, color="b")
    plt.hist(img.flatten(), 256, [0, 256], color="r")
    plt.legend(("cdf", title), loc="upper left")
    plt.savefig("resultados/histo_{}.png".format(title))

img = cv2.imread("ccf394.jpg")
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Obtendo o CDF do histograma e o plotando.
cdf = hist.cumsum()
plot(img, cdf, "before")

# Normalizando o CDF e o plotando.
cdf_normalized = cdf * hist.max() / cdf.max()
plot(img, cdf_normalized, "after")

img = cv2.medianBlur(img, 7)

lower_color_BGR = np.array([115, 165, 165])
upper_color_BGR = np.array([185, 235, 235])

# lower_black_BGR = np.array([0, 0, 0])
# upper_black_BGR = np.array([10, 10, 10])

maskBGR = cv2.inRange(img, lower_color_BGR, upper_color_BGR)
# blackMask = cv2.inRange(res, lower_black_BGR, upper_black_BGR)

result = cv2.bitwise_or(img, img, mask=maskBGR)


#APLICANDO K Means
K = 2
Z = result.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
res = res.reshape((result.shape))
#filtro de borda
edge_canny = cv2.Canny(res, 150, 240)
cv2.imshow("Resultados das bordas", edge_canny)
cv2.imshow("imagem limpa", result)
cv2.imshow("Resultado com apenas 1 tom de cor", res)
print("O histograma está salvo na pasta resultados")
cv2.waitKey();
cv2.destroyAllWindows();
