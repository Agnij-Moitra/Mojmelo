from mojmelo.SVM import SVM_Dual
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import accuracy_score
from python import Python

def main():
    Python.add_to_path(".")
    svmd_test = Python.import_module("SVMD_test")
    data = svmd_test.get_data() # X_train, X_test, y_train, y_test
    svmd = SVM_Dual(kernel = 'rbf')
    svmd.fit(Matrix.from_numpy(data[0]), Matrix.from_numpy(data[2]).T())
    y_pred = svmd.predict(Matrix.from_numpy(data[1]))
    print("SVM_Primal classification accuracy:", accuracy_score(data[3], y_pred))
    svmd_test.test(data[0], data[2], svmd.alpha.T().to_numpy(), svmd.sigma, svmd.bias)
