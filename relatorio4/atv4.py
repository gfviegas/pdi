# -*- coding: utf-8 -*-
# Código do Iris Data set da referencia
#
# Referência:
# https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

# Carregando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import cv2
### CONSTANTES

# 20% dos dados vão pra testes, 80% pra treinamento
TEST_SIZE = 0.2
# Número máximo de iterações
MAX_ITER = 1000
###
def segmentaImagem(frame):
    newFrame = np.reshape(frame, (-1, 3))
    #print(newFrame)
    mask = mlp.predict(newFrame)
    print("=====MASK=====")
    print(mask)
    img = cv2.bitwise_and(newFrame, newFrame, mask)
    print(img)
    return img
###

# URL do dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
caminho = "cores.txt"
# Nome das colunas pro dataset
names = ['blue', 'green', 'red', 'color']
# Carrega o csv externo com os dados do iris
irisdata = pd.read_csv(caminho, names=names)

# Pré-processamento
# Primeiras 4 colunas pra variável X
X = irisdata.iloc[:, 0:3]

# Tipos de dados
Y = irisdata.select_dtypes(include=[object])

# Encoder de labels
le = LabelEncoder()
Y = Y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)

# Scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_test = pd.read_csv("teste.txt")

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Classificador
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=MAX_ITER)
mlp.fit(X_train, y_train.values.ravel())


# Abre um vídeo gravado em disco
video = cv2.VideoCapture('videocanetas.mp4')
outputFile = 'resultados/saidaAtiv3_avi'
(sucesso, frame) = video.read()

# Criando stream de output pro video de resultado
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
outputVideo = cv2.VideoWriter(outputFile, fourcc, 60,
                             (frame.shape[1], frame.shape[0]))

numero = 1
for i in range(0, 2):
    # read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = video.read()
    if not sucesso:  # Final do vídeo
        break

    frame = segmentaImagem(frame)

    # Grava um frame como imagem
    cv2.imwrite('resultados/ativ3_video{}.jpg'.format(numero), frame)
    numero = numero+1

    outputVideo.write(frame)
    cv2.imshow("Exibindo video", frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

outputVideo.release()
video.release()
cv2.destroyAllWindows()
