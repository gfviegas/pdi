# Estratégia: usar KNN. Se tiver mt lento, usar SVM.
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

class LetterClassifier(object):
    trained = False

    def __init__(self, type='svm'):
        super(LetterClassifier, self).__init__()

        if (type == 'knn'):
            self.classifier = KNeighborsClassifier()
        else:
            self.classifier = svm.SVC(kernel='linear', probability=True)

    def train(self, x, y, nColumns):
        self.n = nColumns
        self.classifier.fit(x, y)
        self.trained = True

    def predict(self, letter):
        if (not self.trained):
            raise Exception('Classificador não foi treinado!')
        return self.classifier.predict(letter)
