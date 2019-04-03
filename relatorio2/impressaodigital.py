# Usando a scipy e scikit para Roberts, Sobel, prewitt, scharr
# executar pip install scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.data import camera
from skimage.filters import roberts, sobel, prewitt

image = cv2.imread('impressaodigital.png', 0)
edge_canny = cv2.Canny(image, 150, 240)
edge_prewitt = prewitt(image)
edge_sobel = sobel(image)
edge_roberts = roberts(image)

fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(8, 4))

ax[0].imshow(edge_canny, cmap=plt.cm.gray)
ax[0].set_title('Canny Edge Detection')
ax[1].imshow(edge_prewitt, cmap=plt.cm.gray)
ax[1].set_title('Prewitt Edge Detection')
ax[2].imshow(edge_roberts, cmap=plt.cm.gray)
ax[2].set_title('Roberts Edge Detection')
ax[3].imshow(edge_sobel, cmap=plt.cm.gray)
ax[3].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')
    plt.tight_layout()
    plt.show()
