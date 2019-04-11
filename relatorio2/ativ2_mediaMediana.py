# Relatório 2 - PDI
# Bruno Marra - 3029
# Gustavo Viegas - 3026
# Heitor Passeado - 3055
#
# Descrição da Atividade:
#
# 1) Para a imagem MENINA, aplicar a equalização do histograma, apresentando a
# imagem e histograma antes e depois da equalização. Descreva o que ocorreu com
# o histograma.
# 2) Para a imagem FILTROMEDIANA, converter a imagem para mono e aplicar o
# filtro de media simples e o filtro de mediana, e analisar os resultados.
# 3) Para a imagem de impressao digital, converter para mono e aplicar filtro
# prewitt, sobel, canny e roberts, e comparar os resultados.
# 4) Para a imagem MicroHematuria, utilizar as técnicas que achar necessária
# para obter uma imagem binarizada mostrando as bordas dos globulos de sangue
# presentes na imagem.
#
# Este arquivo se refere a atividade 2).

# Carregando as bibliotecas necessárias
import cv2
import numpy as np

"""
Ruido de Salt N' Pepper
"""
def salt_n_pepper(img, pad, show=1):
    noise = np.random.randint(pad, size=(img.shape[0], img.shape[1], 1))
    img = np.where(noise == 0, 0, img)
    img = np.where(noise == (pad-1), 1, img)
    img = np.uint8(img)
    return img

# Carrega a imagem original
imagemoriginal = cv2.imread("filtromediana.png")
image = salt_n_pepper(imagemoriginal, 300)

# Aplica os respectivos filtros
median = cv2.medianBlur(image, 5)
kernel = np.ones((5, 5), np.float32) / 25
average = cv2.filter2D(image, -1, kernel)

# Salva a imagem
cv2.imwrite("resultados/mediana.png", median)
cv2.imwrite("resultados/media.png", average)
