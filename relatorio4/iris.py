# Código do Iris Data set da referencia
#
# Referência:
# https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

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
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=MAX_ITER)
mlp.fit(X_train, y_train.values.ravel())

# Predições
predictions = mlp.predict(X_test)

# Imprimindo resultados
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
