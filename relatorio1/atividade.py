# -*- coding: utf-8 -*-
# Relatório 1 - PDI
# Bruno Marra - 3029
# Gustavo Viegas - 3026
# Heitor Passeado - 3055
#
# Descrição da Atividade:
#
# Carregar ass imagens imagemCEDAF e imagemCancun. e realizar os seguintes passos:
# 1) separar as imagens na banda bgr
# 2) binarizar cada imagem com o algoritmo de Otsu
# 3) converter a imagem original para cinza e binarrizar com Otsu
# 4) Compare os resultados para as duas imagens
# 5) Aplique Kmeans com k=4..8 para as duas imagens. Escolha um valor de k e,
# para esta imagem segmentada, calcule a área de mata, água e terrra (imagemCEDAF)
# e mar, mata e área construída (prédios) para a imagem Cancun.
# Considere que cada pixel vale 0,5m .
#

# Carregando as bibliotecas necessárias
import numpy as np
import cv2
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sys import maxsize

# Tamanho em metros de cada pixel
TAM_PIXEL = 0.5

# Caminho na pasta local das imagens a serem abertas
images = ['imagemCedaf.png', 'imagemCancun.png']

# Aguarda o usuário para não abrir muitas janelas ao mesmo tempo
def waitAndClose():
    cv2.waitKey()
    cv2.destroyAllWindows()

# Dado uma corSrc, retorna a cor mais pareida na lista de cores coresDest
def categorizacaoCor(corSrc, coresDest):
    melhorScore = maxsize
    melhorCor = [0, 0, 0]

    for (nome, cor) in coresDest.items():
        scoreB = abs(corSrc[0] - cor[0])
        scoreG = abs(corSrc[1] - cor[1])
        scoreR = abs(corSrc[2] - cor[2])
        total = scoreB + scoreG + scoreR

        if (total < melhorScore):
            melhorScore = total
            melhorCor = nome

    return melhorCor

# Após uma operação K-Mean, agrupa as cores parecidas com uma lista de
# coresDesejadas, salvando a quantidade de pixeis dessas coresDesejadas,
# labels do K-Mean e tamanho em metros quadrados
def agrupamentoDetalhes(center, label, tamanhos, indexes, coresDesejadas):
    for (index, centro) in enumerate(center):
        corMaisParecida = categorizacaoCor(centro, coresDesejadas)
        pixeisCorMaisParecida = list(filter(lambda l: l == index, label))
        quantidadePixeis = len(pixeisCorMaisParecida)
        tamanhos[corMaisParecida] = quantidadePixeis * TAM_PIXEL
        indexes[corMaisParecida].append(index)

# Colore de vermelho forte um detalhe da imagem já separado pelo K-Means
def mostraDetalhe(detalhe, indexes, center, label, orig):
    clone = center.copy()
    for x in indexes[detalhe]:
        clone[x] = [0, 0, 255]

        result = clone[label.flatten()]
        result = result.reshape((orig.shape))
        cv2.imshow(detalhe, result)

for image in images:
    # Carregando a imagem
    orig = cv2.imread(image)

    # Separando as bandas BGR
    (B, G, R) = cv2.split(orig)
    bandas = {'Azul': B, 'Verde': G, 'Vermelho': R}

    # Mostrando imagem original e as bandas BGR
    cv2.imshow('Original', orig)
    cv2.imshow('Blue', B)
    cv2.imshow('Green', G)
    cv2.imshow('Red', R)
    waitAndClose()

    for (nome, banda) in bandas.items():
        # Binarizando as bandas da imagem original com o algoritmo de Otsu
        ret, imgf = cv2.threshold(banda, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Banda {} original Otsu'.format(nome), imgf)

    waitAndClose()

    # Convertendo a imagem original pra cinza
    grayImage = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # Binarizando a imagem em cinza com o algoritmo de Otsu
    retGray, imgfGray = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mostrando a imagem em tom de cinza e a imagem em tom de cinza otsu
    cv2.imshow('Imagem em Cinza', grayImage)
    cv2.imshow('Imagem em Cinza Otsu', imgfGray)
    waitAndClose()

    # Aplicando o K-Means para 4..8
    Z = orig.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    for K in range(4, 9):
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)

        res = center[label.flatten()]
        res = res.reshape((orig.shape))
        cv2.imshow('K-Means {}'.format(K), res)

        if (K == 6 and image == 'imagemCedaf.png'):
            coresDesejadas = {
                'mato': [54, 47, 31], # Mato
                'agua': [86, 85, 60], # Água
                'terra': [119, 122, 124] # Terra
            }

            indexes = {
                'mato': [], # Mato
                'agua': [], # Água
                'terra': [] # Terra
            }

            tamanhos = {
                'mato': 0, # Mato
                'agua': 0, # Água
                'terra': 0 # Terra
            }

            agrupamentoDetalhes(center, label, tamanhos, indexes, coresDesejadas)

            mostraDetalhe('mato', indexes, center, label, orig)
            mostraDetalhe('agua', indexes, center, label, orig)
            mostraDetalhe('terra', indexes, center, label, orig)
            waitAndClose()

            # Imprime os tamanhos de cada área em metros quadrados.
            print(tamanhos)
        elif (K == 4 and image == 'imagemCancun.png'):
            coresDesejadas = {
                'mar': [138, 98, 49], # Mar
                'floresta': [63, 68, 35], # Mato
                'area': [115, 127, 126] # Área Construida
            }

            indexes = {
                'mar': [], # Mar
                'floresta': [], # Mato
                'area': [] # Área Construida
            }

            tamanhos = {
                'mar': 0, # Mar
                'floresta': 0, # Mato
                'area': 0 # Área Construida
            }

            agrupamentoDetalhes(center, label, tamanhos, indexes, coresDesejadas)

            mostraDetalhe('mar', indexes, center, label, orig)
            mostraDetalhe('floresta', indexes, center, label, orig)
            mostraDetalhe('area', indexes, center, label, orig)
            waitAndClose()

            # Imprime os tamanhos de cada área em metros quadrados.
            print(tamanhos)

    waitAndClose()
