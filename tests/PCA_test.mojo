from mojmelo.PCA import PCA   
from mojmelo.utils.Matrix import Matrix 
from python import Python

def main():
    Python.add_to_path(".")
    pca_test = Python.import_module("PCA_test")
    data = pca_test.get_data() # X, y
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(Matrix.from_numpy(data[0]))
    X_projected = pca.transform(Matrix.from_numpy(data[0]))
    print("Shape of X:", data[0].shape)
    print("Shape of transformed X:", '(' + str(X_projected.height) + ', ' + str(X_projected.width) + ')')
    pca_test.test(X_projected.to_numpy(), data[1])
