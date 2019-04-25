# Relatório 4 - PDI
# Bruno Marra - 3029
# Gustavo Viegas - 3026
# Heitor Passeado - 3055
#
# Descrição da Atividade:
#
# 1) A partir do codigo apresentado em sala de aula, realize 4 alterações
# (altere tamanho da rede, numero iteraçoes, taxa de aprendizado, etc) e analise
# os resultados.
# 2) a partir do código apresentado em sala, utilize um outro dataset. Procure
# um dataset com mais amostras do que o iris dataset, e execute o codigo,
# analisando os resultados
# 3) A partir dos filmes das canetas apresentados na aula anterior, escolha uma
# das cores para ser segmentada, para gerar um filme com apenas aquela cor se
# movimentando. Tire pelo menos 50 amostras de cada região a ser segmentada,
# gere um filme final.
#
# Este arquivo se refere a atividade 3).
#
# Referência:
# https://www.kaggle.com/c/forest-cover-type-prediction/data

# Carregando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Desligando warnings
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

### CONSTANTES
# % dos dados vão pra testes, o restante pra treinamento
TEST_SIZE = 0.05
# Número máximo de iterações
MAX_ITER = 5000
###

# Nome das colunas pro dataset
names = ['blue', 'green', 'red', 'color']
# Carrega o csv externo com os dados do iris
df = pd.read_csv('cores.csv', names=names)

# Pré-processamento
# Primeiras 4 colunas pra variável X
X = df.iloc[:, 0:3]

# Tipos de dados
Y = df.select_dtypes(include=[object])

# Encoder de labels
le = LabelEncoder()
Y = Y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)

# Scaler
scaler = StandardScaler()
scaler.fit(X_train)

# X_test = pd.read_csv('teste.csv')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Classificador
mlp = MLPClassifier(hidden_layer_sizes=(6), max_iter=MAX_ITER)
mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)
print(classification_report(y_test,predictions))

# Segmenta a Imagem
def segmentaImagem(frame):
    # Converte o frame pra um array de RGBs
    # newFrame = np.asarray(frame)
    newFrame = frame.reshape((-1, 3))
    # Prediz a classe do frame em formato de array
    mask = mlp.predict(newFrame)
    # Converte o array pra uma matriz novamente
    width, height, depth = frame.shape
    mask = mask.reshape((width, height)).astype(np.uint8)

    return cv2.bitwise_and(frame, frame, mask=mask)


# Abre um vídeo gravado em disco
video = cv2.VideoCapture('videocanetas.mp4')
outputFile = 'resultados/saidaAtiv3.avi'
(sucesso, frame) = video.read()

# Criando stream de output pro video de resultado
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
outputVideo = cv2.VideoWriter(outputFile, fourcc, 60,
                             (frame.shape[1], frame.shape[0]))

numero = 0
while True:
    # read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = video.read()
    if not sucesso:  # Final do vídeo
        break
    # if numero == 242:
    #     cv2.imshow('Image', frame)
    #     cv2.waitKey()
    frame = segmentaImagem(frame)

    # Grava um frame como imagem
    if (numero % 20 == 0 and numero < 1000):
        cv2.imwrite('resultados/ativ3_video{}.jpg'.format(numero), frame)
    numero += 1

    outputVideo.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

outputVideo.release()
video.release()
cv2.destroyAllWindows()
