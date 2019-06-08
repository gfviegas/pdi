# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from string import ascii_lowercase

from .classifier import LetterClassifier

def writeLetter(letter):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height) = cv2.getTextSize(letter, font, 6, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = frame.shape[0] - 25
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

    cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
    cv2.putText(frame, letter, (text_offset_x, text_offset_y), font, 6, (255,255,255), 3)
    return

def treatImage(frame):
    frame = cv2.resize(frame, (480, 640))

    # aplica filtro gaussiano e coloca imagem preto e branco
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.GaussianBlur(img_grey, (5,5), 0)

    # aplica otsu para binarizar
    ret, img_grey = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #tira o contorno da imagem
    contours, heirarchy = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borderImg = np.zeros((480,640,3), np.uint8)
    cv2.drawContours(borderImg,contours,-1,(125,125,0),1)
    borderGrey = cv2.cvtColor(borderImg, cv2.COLOR_BGR2GRAY)

    # Calculate Moments
    moment = cv2.moments(borderGrey)

    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moment)
    print(huMoments)
    cv2.imshow("Imagem capturada", borderImg)
    return huMoments

def videoLive():
    cap = cv2.VideoCapture(0)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    framecount = 0

    while(1):
        ret, frame=cap.read()
        frame = cv2.flip(frame, 1)
        framecount += 1

        # Check if this is the frame closest to 5 seconds
        if framecount == (framerate * 5):
            framecount = 0
            writeLetter('B')

        treatImage(frame)
            #cv2.imshow('Imagem capturada', frame)

          # TODO
          # mandar o frame para a função de avaliação

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Iniciando Classificador
LC = LetterClassifier()

def applyPCA(data):
    pca = PCA(n_components=0.90, svd_solver='full')
    pca.fit_transform(data)

def trainClassifier():
    # df = pd.DataFrame([[]], columns=['n sei', 'nsei', 'letter'])
    df = pd.DataFrame()

    for letter in ascii_lowercase:
        for i in range(10):
            imgPath = 'alphabet/{}{}.jpg'.format(letter, i)
            data = [] # Chamar o método que retorna os momentos
            applyPCA(data)
            data.insert(0, letter)
            df.append([data])


    # A primeira coluna será chamada de letter já q inserimos a letter nela
    df.rename(columns={ df.columns[0]: 'letter' })

    # Dados pra treinamento
    x = df.drop('letter', axis=1)
    y = df['letter']

    # Treina o classificador
    LC.train(x, y)


if __name__ == '__main__':
    trainClassifier()
    videoLive()
