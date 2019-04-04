import cv2
import numpy as np


def salt_n_pepper(img, pad, show=1):
    noise = np.random.randint(pad, size=(img.shape[0], img.shape[1], 1))
    img = np.where(noise == 0, 0, img)
    img = np.where(noise == (pad-1), 1, img)
    img = np.uint8(img)
    return img


# Carrega a imagem original
imagemoriginal = cv2.imread('filotromediana.png')
image = salt_n_pepper(imagemoriginal, 300)

# Aplica os respectivos filtros
median = cv2.medianBlur(image, 5)

kernel = np.ones((5, 5), np.float32) / 25
average = cv2.filter2D(image, -1, kernel)

# Escreve o nome de cada filtro na imagem correspondente
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(median, 'Median', (60, 320), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Salva a imagem
cv2.imwrite("mediana.png", median)
cv2.imwrite("media.png", average)
