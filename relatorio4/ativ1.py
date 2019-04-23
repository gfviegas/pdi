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
# Este arquivo se refere a atividade 1).
#
# Referência:
# https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# Carregando as bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

### CONSTANTES

# 20% dos dados vão pra testes, 80% pra treinamento
TEST_SIZE = 0.25
# Número máximo de iterações
MAX_ITER = 1000
# Tamanho das camadas escondidas. No caso, (10, 10, 10) = 3 camadas de 10 nós
LAYERS_SIZE = (10, 10, 10)

###

# URL do dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Nome das colunas pro dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Carrega o csv externo com os dados do iris
irisdata = pd.read_csv(url, names=names)

# Pré-processamento
# Primeiras 4 colunas pra variável X
X = irisdata.iloc[:, 0:4]
# Tipos de dados
Y = irisdata.select_dtypes(include=[object])

# Encoder de labels
le = LabelEncoder()
Y = Y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)

# Scaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Classificador
# mlp = MLPClassifier(hidden_layer_sizes=LAYERS_SIZE, max_iter=MAX_ITER)
mlp = MLPClassifier(
    activation='relu',
    alpha=0.0001,
    batch_size='auto',
    beta_1=0.9,
    hidden_layer_sizes=LAYERS_SIZE,
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=MAX_ITER,
    momentum=0.9,
    nesterovs_momentum=True,
    power_t=0.5,
    random_state=None,
    shuffle=True,
    solver='adam',
    tol=0.0001,
    validation_fraction=0.1,
    verbose=False,
    warm_start=False
)
mlp.fit(X_train, y_train.values.ravel())

# Predições
predictions = mlp.predict(X_test)

# Imprimindo resultados
print(predictions)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
