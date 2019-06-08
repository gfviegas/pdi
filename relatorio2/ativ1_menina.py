# -*- coding: utf-8 -*-
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
# Este arquivo se refere a atividade 1).
#
# Referência:
# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

# Carregando as bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt


"""
Plota o histograma cdf de uma imagem
"""
def plot(img, cdf, title):
    plt.plot(cdf, color="b")
    plt.hist(img.flatten(), 256, [0, 256], color="r")
    plt.legend(("cdf", title), loc="upper left")
    plt.savefig("resultados/histo_{}.png".format(title))

# Carregando a imagem da menina e gerando o histograma.
img = cv2.imread("menina.jpg")
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Obtendo o CDF do histograma e o plotando.
cdf = hist.cumsum()
plot(img, cdf, "before")

# Normalizando o CDF e o plotando.
cdf_normalized = cdf * hist.max() / cdf.max()
plot(img, cdf_normalized, "after")
