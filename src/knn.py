import numpy as np
class KNN:
    def __init__(self, k=5, task="classification", distance="euclidean"):
        #construtor
        self.k = k
        self.task = task
        self.distance = distance
    def fit(self, X_train, y_train): #treinamento
        self.X_train = X_train
        self.y_train = y_train
    def calculate_distances(self, x1, x2):
        if self.distance == "manhattan": #manhattan
            return np.sum(np.abs(x1-x2))
        return np.sqrt(np.sum((x1-x2)**2)) #euclidiana
    def calculate_predicts(self, x):
        distances = [self.calculate_distances(x, x_train)
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_label = [self.y_train[i] for i in k_indices]
        if self.task == "classification":
             unique, counts = np.unique(k_nearest_label, return_counts=True)
             return unique[np.argmax(counts)]
        if self.task == "regression":
             return np.mean(k_nearest_label)
        else:
             raise ValueError("Tarefa sera classification ou regression")
    def predict(self, X_test): #predicao
         predictions = [self.calculate_predicts(x) for x in X_test]
         return predictions
