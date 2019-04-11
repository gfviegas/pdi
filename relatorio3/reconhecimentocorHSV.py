# -*- coding: utf-8 -*- 
# segmenta cores no espaço hsv , detectando a cor azul.
# funciona com camera ou imagem lida.

import cv2
import numpy as np

#Dica
# Region Of Interest of a Image (ROI)
# se usarmos o comando abaixo, estamos extraindo uma janela de uma imagem original
# e salvando em temp.
# Depois, substituimos na imagem original uma outra janela com a imagem copiada,
# criando assim um efeito de repetição da imagem recortada na imagem original

# temp = img[280:340, 330:390]
# img[273:333, 100:160] = temp

# isto é muito util, pois podemos criar imagens menores com regiao de nosso
# interesse , a parrtir de imagens maiores. O comando abaixo cria uma nova imagem
# de zeros, de 300x200. Se recortarmos um pedaço da imagem original, podemos
# colar nesta noma "imagem"
# mask = np.zeros((300,200), np.uint8)



def usaimagemlida():
    img=cv2.imread("caneta.jpg")
    #como a foto foi tirada do celular, vamos alterar o tamanho da imagem
    print('Original Dimensions : ',img.shape)

    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    print('Dimensoes após redução de escala : ',resized.shape)

    cv2.imshow("Imagem Reduzida", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    img=resized.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV e RGB
    # estes valores de pixels normalmente são amostrados de imagens
    # conhecidas e utilizados para segmentar imagens novas.

    lower_blue_HSV = np.array([110,50,50])
    upper_blue_HSV = np.array([130,255,255])
    lower_blue_BGR = np.array([100,30,0])
    upper_blue_BGR = np.array([200,100,60])
    # Threshold the HSV image to get only blue colors
    maskHSV = cv2.inRange(hsv, lower_blue_HSV, upper_blue_HSV)
    maskBGR = cv2.inRange(img, lower_blue_BGR, upper_blue_BGR)
    # Bitwise-AND mask and original image
    resHSV = cv2.bitwise_and(img,img, mask= maskHSV )
    resRGB = cv2.bitwise_and(img,img, mask= maskBGR )
    cv2.imshow('Imagem Original',img)
    cv2.imshow('Imagem HSV',hsv)
    cv2.imshow('Mascara com HSV',maskHSV)
    cv2.imshow('Resultado com HSV',resHSV)
    cv2.imshow('Mascara com RGB',maskBGR)
    cv2.imshow('Resultado com RGB',resRGB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
if (cap.isOpened()== False):
  print("Erro abertura da camera")
  usaimagemlida()
else:
    while(True):

        # Take each frame
        ret , frame = cap.read()
        if ret == True:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # define range of blue color in HSV
            lower_blue = np.array([110,50,50])
            upper_blue = np.array([130,255,255])

            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame,frame, mask= mask)

            cv2.imshow('frame',frame)
            cv2.imshow('mask',mask)
            cv2.imshow('res',res)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter('saida.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                cap.release()
                break


    cv2.destroyAllWindows()
