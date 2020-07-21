from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, recall_score, precision_score
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32')


X_train = X_train / 255
X_test = X_test / 255

knn = KNeighborsClassifier(n_neighbors=3)
X_train = X_train[:3000]
y_train = y_train[:3000]
print(X_train.shape)
print(y_train.shape)

knn.fit(X_train, y_train)
print(f'FIT END')
X_test = X_test[:1000]
y_test = y_test[:1000]
print(f'Accuracy {accuracy_score(knn.predict(X_test), y_test)}')
print(f'Recall {recall_score(knn.predict(X_test), y_test, average="macro")}')
print(f'Precision {precision_score(knn.predict(X_test), y_test, average="macro")}')