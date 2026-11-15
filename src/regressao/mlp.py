import numpy as np
class MLP: #Multi Layer Perceptron para regressao
    def __init__(self, hidden_layers=(16,), activation="tanh", learning_rate=0.01,
                 n_epochs=100, batch_size=32, random_state=42):
        #construtor
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
    def fit(self, X_train, y_train): #treinamento
        rng = np.random.default_rng(self.random_state)
        N = X_train.shape[0]
        #alvo padronizado: a saida linear parte de zero, perto da media
        self._y_mean, self._y_std = np.mean(y_train), np.std(y_train)
        d = ((y_train - self._y_mean) / self._y_std).reshape(-1, 1)
        sizes = [X_train.shape[1], *self.hidden_layers, 1]
        self.W = [rng.normal(0, np.sqrt(1 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        for _ in range(self.n_epochs):
            indices = rng.permutation(N) #embaralha as amostras a cada epoca
            for start in range(0, N, self.batch_size):
                batch = indices[start:start + self.batch_size]
                A = self.forward(X_train[batch])
                #gradiente do erro quadratico medio na saida (linear)
                delta = 2 * (A[-1] - d[batch]) / len(batch)
                for l in reversed(range(len(self.W))): #backpropagation
                    grad_W = A[l].T @ delta
                    grad_b = np.sum(delta, axis=0)
                    if l > 0: #propaga o erro para a camada anterior
                        delta = (delta @ self.W[l].T) * self.derivative(A[l])
                    self.W[l] -= self.learning_rate * grad_W
                    self.b[l] -= self.learning_rate * grad_b
    def forward(self, X): #saida de cada camada
        A = [X]
        for l in range(len(self.W)):
            u = A[-1] @ self.W[l] + self.b[l]
            last = l == len(self.W) - 1
            A.append(u if last else self.activate(u)) #saida sem ativacao
        return A
    def activate(self, u):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-u))
        if self.activation == "tanh":
            return np.tanh(u)
        if self.activation == "relu":
            return np.maximum(0, u)
        raise ValueError("Ativacao sera sigmoid, tanh ou relu")
    def derivative(self, a): #derivada escrita em funcao da saida da camada
        if self.activation == "sigmoid":
            return a * (1 - a)
        if self.activation == "tanh":
            return 1 - a ** 2
        return (a > 0).astype(float) #relu
    def predict(self, X_test): #predicao
        y = self.forward(X_test)[-1].ravel()
        return y * self._y_std + self._y_mean
