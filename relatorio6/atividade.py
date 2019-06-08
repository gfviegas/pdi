# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from mahotas.features import zernike_moments
from sklearn.decomposition import PCA
from string import ascii_lowercase

from classifier import LetterClassifier

# Iniciando Classificador
LC = LetterClassifier(type='knn')


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
    frame = cv2.resize(frame, (120, 120))

    # aplica filtro gaussiano e coloca imagem preto e branco
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.GaussianBlur(img_grey, (5,5), 0)

    # aplica otsu para binarizar
    ret, img_grey = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #tira o contorno da imagem
    contours, heirarchy = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borderImg = np.zeros((120,120,3), np.uint8)
    cv2.drawContours(borderImg,contours,-1,(125,125,0),1)
    borderGrey = cv2.cvtColor(borderImg, cv2.COLOR_BGR2GRAY)

    # Calculate Moments
    moment = cv2.moments(borderGrey)

    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moment)

    # Calcula momento de Zernike
    zeMoments = zernike_moments(borderGrey, radius=1)

    # cv2.imshow("Imagem capturada", borderImg)
    return huMoments + zeMoments

def videoLive():
    cap = cv2.VideoCapture(0)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    framecount = 0

    frameData = list()

    while(1):
        ret, frame=cap.read()
        frame = cv2.flip(frame, 1)
        framecount += 1

        if (len(frameData) < 5):
            frameData.append(frame)
        else:
            letter = predictLetter(frameData)
            writeLetter(letter)
            frameData = list()

        # Check if this is the frame closest to 5 seconds
        # if framecount == (framerate * 5):
        #     framecount = 0
        #     letter = predictLetter(frame)
        #     writeLetter(letter)

        cv2.imshow('Imagem capturada', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def applyPCA(data):
    # print('PRE-PCA: ', len(data), len(data[0]))
    pca = PCA(n_components=0.9)
    data = pca.fit_transform(data)
    # print('POS-PCA: ', len(data), len(data[0]))
    # print('Variancia: ', pca.explained_variance_)
    # print('Variancia: ', pca.explained_variance_ratio_)
    # print('Componentes: ', pca.components_)
    return data

def trainClassifier():
    dfData = list()
    nComp = 0

    for letter in ascii_lowercase:
        for i in range(10):
            # Abre a imagem
            imgPath = 'alphabet/{}{}.jpg'.format(letter, i)
            img = cv2.imread(imgPath)

            # Extrai os dados do momento
            data = treatImage(img).ravel().tolist()
            data.insert(0, letter)
            dfData.append(data)
        # for entry in data:
        #     nComp = max(nComp, len(entry))
        #
        #     # Insere no dataframe
        #     entry.insert(0, letter)
        #     dfData.append(entry)

    # Aplica a PCA
    dataWithoutLetter = list(map(lambda d: d[1:], dfData))
    data = applyPCA(dataWithoutLetter).tolist()
    nComp = len(data[0])

    columns = ['c{}'.format(i) for i in range(nComp)]
    columns.insert(0, 'letter')

    for i in range(len(dfData)):
        data[i].insert(0, dfData[i][0])

    df = pd.DataFrame(data, columns=columns).fillna(0)

    print(df.info())
    print(df.head(100))

    # Dados pra treinamento
    x = df.drop('letter', axis=1)
    y = df['letter']

    # Treina o classificador
    LC.train(x, y, nComp)

def predictLetter(frameList):
    frameData = list()
    for frame in frameList:
        data = treatImage(frame).ravel()
        frameData.append(data)


    pca = PCA(LC.n)
    data = pca.fit_transform(frameData).tolist()

    columns = ['c{}'.format(i) for i in range(LC.n)]
    df = pd.DataFrame(data, columns=columns).fillna(0)

    predictions = LC.predict(df)
    print(predictions)

    # data = treatImage(img).ravel().reshape(-1, 1)
    # print(data[:1])
    # return LC.predict(data[:1])

if __name__ == '__main__':
    trainClassifier()

    print(predictLetter([cv2.imread('alphabet/g{}.jpg'.format(i)) for i in range(6) ]))
    print(predictLetter([cv2.imread('alphabet/a{}.jpg'.format(i)) for i in range(6) ]))
    print(predictLetter([cv2.imread('alphabet/w{}.jpg'.format(i)) for i in range(6) ]))
    # videoLive()
