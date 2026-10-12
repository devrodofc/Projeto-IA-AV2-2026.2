import numpy as np
class NaiveBayes: #Univariado
    def fit(self,X_train, y_train): #treinamento da maquina
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var =  np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._mean[idx, :]  = X_c.mean(axis=0)
            self._var[idx, :]  = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0]/float(n_samples)
    def predict(self, X_test): #predicao
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[idx]) #log evita underflow
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + posterior)
        return self._classes[np.argmax(posteriors)]
    def _pdf(self, class_idx, x): #densidade normal univariada
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
