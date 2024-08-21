from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_data():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )
    return[X_train, X_test, y_train, y_test]
