import numpy as np
class Perceptron: #Simples
    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        #construtor
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
    def fit(self, X_train, y_train): #treinamento
        rng = np.random.default_rng(self.random_state)
        self._classes = np.unique(y_train)
        d = (y_train == self._classes[1]).astype(int) #saida desejada: 0 ou 1
        N = X_train.shape[0]
        X_train = np.column_stack((np.ones(N), X_train)) #coluna de 1s: bias
        #pesos iniciais aleatorios (com zeros a taxa de aprendizado nao teria efeito)
        self.w = rng.uniform(-0.5, 0.5, X_train.shape[1])
        for _ in range(self.n_epochs):
            errors = 0
            for i in rng.permutation(N): #embaralha as amostras a cada epoca
                y = self.activation(X_train[i] @ self.w)
                e = d[i] - y
                if e != 0:
                    self.w += self.learning_rate * e * X_train[i] #regra de aprendizado
                    errors += 1
            if errors == 0: #convergiu: nenhuma amostra errada na epoca
                break
    def activation(self, u): #funcao degrau
        return np.where(u >= 0, 1, 0)
    def predict(self, X_test): #predicao
        N = X_test.shape[0]
        X_test = np.column_stack((np.ones(N), X_test))
        return self._classes[self.activation(X_test @ self.w)]
