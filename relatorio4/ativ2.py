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
from sklearn.metrics import classification_report, confusion_matrix

### CONSTANTES
# 20% dos dados vão pra testes, 80% pra treinamento
TEST_SIZE = 0.2
# Número máximo de iterações
MAX_ITER = 1000
# Tamanho das camadas escondidas. No caso, (10, 10, 10) = 3 camadas de 10 nós
LAYERS_SIZE = (10, 10, 10)
###

# URL do dataset - Floresta (ver referência)
# O arquivo train.csv tem a classificação de Cover Type, o test.csv não.
url = 'forest/train.csv'
# Carrega o csv com os dados da floresta (df = Pandas DataFrame)
df = pd.read_csv(url)
# Removendo a coluna de ID, que é desnecessário pro nosso escopo do trabalho
df.drop('Id', axis=1, inplace=True)

## Pré-processamento
# A variável X vai ter todos os dados, menos a coluna Cover_Type, que é a que
# queremos predizer.
X = df.drop('Cover_Type', axis=1)
# A variável Y vai ter os dados do Cover Type que queremos prever
Y = df['Cover_Type']


## Encoder de labels
# Não precisamos aplicar o LabelEncoder porque o CoverType já é do tipo inteiro
# le = LabelEncoder()
# Y = Y.apply(le.fit_transform)

# Pra evitar over-fitting, dividiremos o dataset em training e test.
# O training é usado pra treinar a rede neural e o teste pra avaliar a
# performance.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE)

## Feature Scaling
# Antes de fazer predições é uma boa prática escalar as features pra que todas
# possam ser uniformemente avaliadas. Só aplicamos isso no training.
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


## Classificador
# Configurando o classificador com as constantes definidas no início do codigo
mlp = MLPClassifier(hidden_layer_sizes=LAYERS_SIZE, max_iter=MAX_ITER)
# Treina o algoritmo com os dados de treino.
mlp.fit(X_train, Y_train.values.ravel())

## Predições
predictions = mlp.predict(X_test)

## Imprimindo resultados
print(predictions)
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
