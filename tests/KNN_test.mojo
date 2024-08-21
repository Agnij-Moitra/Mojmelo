from mojmelo.KNN import KNN
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    knn_test = Python.import_module("KNN_test")
    data = knn_test.get_data() # X_train, X_test, y_train, y_test
    knn = KNN(k = 3)
    knn.fit(Matrix.from_numpy(data[0]), data[2])
    y_pred = knn.predict(Matrix.from_numpy(data[1]))
    print("KNN classification accuracy:", accuracy_score(data[3], y_pred))
