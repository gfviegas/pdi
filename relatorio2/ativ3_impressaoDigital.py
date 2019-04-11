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
# Este arquivo se refere a atividade 3).

# Carregando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.data import camera
from skimage.filters import roberts, sobel, prewitt

# Lê a imagem de entrada e aplica os algoritmos edges
image = cv2.imread("impressaodigital.png", 0)
edge_canny = cv2.Canny(image, 150, 240)
edge_prewitt = prewitt(image)
edge_sobel = sobel(image)
edge_roberts = roberts(image)

# Faz os subplots pra cada filtro
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(6, 6))

# Pra cada filtro, exibe a imagem com o filtro
ax1.set_title("Canny Edge Detection")
ax1.imshow(edge_canny, cmap=plt.cm.gray)
ax1.axis("off")

ax2.set_title("Prewitt Edge Detection")
ax2.imshow(edge_prewitt, cmap=plt.cm.gray)
ax2.axis("off")

ax3.set_title("Roberts Edge Detection")
ax3.imshow(edge_roberts, cmap=plt.cm.gray)
ax3.axis("off")

ax4.set_title("Sobel Edge Detection")
ax4.imshow(edge_sobel, cmap=plt.cm.gray)
ax4.axis("off")

# Saída dos resultados
plt.tight_layout()
plt.savefig("resultados/impressaoDigital_resuts.png")
