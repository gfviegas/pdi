# Relatório 3 - PDI
# Bruno Marra - 3029
# Gustavo Viegas - 3026
# Heitor Passeado - 3055
#
# Descrição da Atividade:
#
# 1) Para cada imagem de cubo(10 imagens), extraia 5 valores de R,G,B,L,a,b,
# extraia e calcule a media e desvio padrão para ver a variação da matiz de cor
# em cada agrupamento. Faça isto para duas cores distintas do cubo.
# Ao final desta operação, voce vai ter , para uma cor, um total de 50 pixels
# amostrados. Calcule novamente a media e desvio padrao desta amostra, para
# verificar se ocorreu muita variação de cor. Analise principalemnte os valores
# de L,a,b. Lembre-se de fazer isto para duas cores do cubo. Utilize a lib numpy
# ou uma planilha eletronica para visualizar os resultados.
#
# 2) Abra arquivo caneta.jpg e escolha 12 pixels amostrados da cor azul, tomando
# cuidado para se extrairem pixels das pontas e meio da imagem. Ache os valores
# maximos e minimos para os pixels, e usando o script reconhecimentoHSV,
# apresente a segmentação obtida nas 4 imagens de caneta, e analise o resultado.
#
# 3) Da mesma forma que realizado no item 2, realize o mesmo procedimento para
# o filme videocaneta, realizando a segmentação para as 4 cores presentes,
# obtendo 4 filmes resultantes. Estes filmes devem ser obtidos aplicando a
# mascara de segmentação obtida, exiba o filme apresentando somente a ponta da
# caneta segmentada se movendo...
#
# 4) Da mesma forma que no item 4, gere imagens do arquivo RUN.AVI, selecione
# somente amostras de pixels dos corredores que usam camisa azul escura, e faça
# o procedimento de segmentação para exibir somente estes corredores.
#
# Este arquivo se refere a atividade 2).

# Carregando as bibliotecas necessárias
import cv2
import numpy as np

img = cv2.imread("images/caneta.jpg")
# Como a foto foi tirada do celular, vamos alterar o tamanho da imagem
print('Dimensões Originais: {}'.format(img.shape))

scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize da imagem
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print('Dimensões após redução de escala: {}'.format(resized.shape))

cv2.imshow("Imagem Reduzida", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = resized.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Valores HSV encontrados
# [[117, 219, 49], [117, 206, 42], [117, 210, 40], [116, 205, 41],
# [117, 210, 34], [118, 191, 36], [114, 222, 31], [116, 226, 35],
# [118, 204, 30], [116, 205, 36], [118, 199, 32], [117, 210, 40]]

# Intervalo HSV
lower_blue_HSV = np.array([110, 150, 20])
upper_blue_HSV = np.array([130, 255, 255])

# Intervalo BGR
lower_blue_BGR = np.array([30, 8, 6])
upper_blue_BGR = np.array([49, 11, 7])

# Máscara pra pegar apenas valores em azul
maskHSV = cv2.inRange(hsv, lower_blue_HSV, upper_blue_HSV)
maskBGR = cv2.inRange(img, lower_blue_BGR, upper_blue_BGR)

# Bitwise AND para mostrar apenas cores azuis na imagem
resHSV = cv2.bitwise_and(img, img, mask=maskHSV)
resRGB = cv2.bitwise_and(img, img, mask=maskBGR)
cv2.imshow('Imagem Original', img)
cv2.imshow('Imagem HSV', hsv)
cv2.imshow('Mascara com HSV', maskHSV)
cv2.imshow('Resultado com HSV', resHSV)
cv2.imshow('Mascara com RGB', maskBGR)
cv2.imshow('Resultado com RGB', resRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()
