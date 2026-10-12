import numpy as np

class MRegression:
    def __init__(self):
        self.beta = None
    def fit(self, X_train, y_train):
        N = X_train.shape[0]
        X_train = np.column_stack((np.ones(N), X_train))
        self.beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        #pseudo inversa (casos sem matriz quadratica, nao possui inversa)
        #Moore Pensore
    def predict(self, X_test):
        N = X_test.shape[0]
        X_test = np.column_stack((np.ones(N), X_test))
        return X_test @ self.beta
