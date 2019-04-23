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
# Este arquivo se refere a atividade 2).
#
# Referência:
# https://www.kaggle.com/c/forest-cover-type-prediction/data

# Carregando as bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

# URL do dataset - Floresta (ver referência)
# O arquivo train.csv tem a classificação de Cover Type, o test.csv não.
url = 'forest/train.csv'
# Carrega o csv com os dados da floresta (df = Pandas DataFrame)
df = pd.read_csv(url)
# Removendo a coluna de ID, que é desnecessário pro nosso escopo do trabalho
df.drop('Id', axis=1, inplace=True)

class NeuralForest(object):
    def __init__(self, TEST_SIZE=None, MAX_ITER=None, LAYERS_SIZE=None):
        if TEST_SIZE is None:
            TEST_SIZE = 0.2
        if MAX_ITER is None:
            MAX_ITER = 1000
        if LAYERS_SIZE is None:
            LAYERS_SIZE = (10, 10, 10)

        self.TEST_SIZE = TEST_SIZE
        self.MAX_ITER = MAX_ITER
        self.LAYERS_SIZE = LAYERS_SIZE

    def loadDataset(self):
        self.df = df.copy()

    def preProcessing(self):
        ## Pré-processamento
        # A variável X vai ter todos os dados, menos a coluna Cover_Type, que é a que
        # queremos predizer.
        self.X = self.df.drop('Cover_Type', axis=1)

        # A variável Y vai ter os dados do Cover Type que queremos prever
        self.Y = self.df['Cover_Type']

    def labelEncode(self):
        ## Encoder de labels
        # Não precisamos aplicar o LabelEncoder porque o CoverType já é do tipo inteiro
        # le = LabelEncoder()
        # Y = Y.apply(le.fit_transform)
        pass

    def datasetSplit(self):
        # Pra evitar over-fitting, se divide o dataset em training e test.
        # O training é usado pra treinar a rede neural e o teste pra avaliar a
        # performance.
        #
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=self.TEST_SIZE)
        pass

    def featureScaling(self):
        ## Feature Scaling
        # Antes de fazer predições é uma boa prática escalar as features pra que todas
        # possam ser uniformemente avaliadas. Só aplicamos isso no training.
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def classify(self):
        ## Classificador
        # Configurando o classificador com as constantes definidas no início do codigo
        self.mlp = MLPClassifier(hidden_layer_sizes=self.LAYERS_SIZE,
                                 max_iter=self.MAX_ITER)
        # Treina o algoritmo com os dados de treino.
        self.mlp.fit(self.X_train, self.Y_train.values.ravel())

    def predictions(self):
        ## Predições
        predictions = self.mlp.predict(self.X_test)
        return (predictions,
                accuracy_score(self.Y_test, predictions),
                classification_report(self.Y_test, predictions),
                confusion_matrix(self.Y_test, predictions)
                )

    def evaluate(self, verbose=False):
        self.loadDataset()
        self.preProcessing()
        self.labelEncode()
        self.datasetSplit()
        self.featureScaling()
        self.classify()

        predictions = self.predictions()
        if (verbose):
            preds, score, report, matrix = predictions
            print(score)
            print(report)
            print(matrix)

        return predictions

# Example
# network = NeuralForest(0.25, 1000, (20, 10, 10))
# network.evaluate(verbose=True)
#
# Uma saida do algoritmo genético mais eficiente...
network = NeuralForest(0.2392683376355929, 1294, (14, 24, 37, 26, 28, 37, 13,
                                                  39))
network.evaluate(verbose=True)
