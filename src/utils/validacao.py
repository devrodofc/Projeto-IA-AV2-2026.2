import numpy as np
import math

def k_fold(X, y, k=5, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho!")
    if k < 2 or k > len(X):
        raise ValueError("k deve estar entre 2 e a quantidade de amostras!")
    n_samples = len(X)
    indices = np.random.permutation(n_samples) #embaralhar os dados
    fold_size = math.ceil(n_samples / k) #tamanho de cada fold
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size],
                                        indices[(i + 1) * fold_size:]))
        if X.ndim == 1:
            X_train, X_test = X[train_indices], X[test_indices]
        else:
            X_train, X_test = X[train_indices,:], X[test_indices,:]
        y_train, y_test = y[train_indices], y[test_indices]
        folds.append((X_train, X_test, y_train, y_test))
    return folds
