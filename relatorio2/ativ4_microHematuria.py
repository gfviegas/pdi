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
# Este arquivo se refere a atividade 4).

# Carregando as bibliotecas necessárias
import numpy as np
import cv2

# Lê a imagem de entrada
img = cv2.imread("MicroHematuria.jpg", cv2.IMREAD_GRAYSCALE)

# Aplicando filtro canny
edge_canny = cv2.Canny(img, 150, 240)
Z = edge_canny.reshape((-1, 3))
Z = np.float32(Z)

# Aplicando K-Means com K = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 1.0)
K = 8
ret, label, center = cv2.kmeans(Z, K, None, criteria, 3,
                                cv2.KMEANS_RANDOM_CENTERS)

# Convertendo estruturas de dados e fazendo imagem original
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((edge_canny.shape))

cv2.imwrite("resultados/microHematuriaBinarizada.jpg", res2)
