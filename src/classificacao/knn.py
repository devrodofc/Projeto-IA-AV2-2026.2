import numpy as np


def distancia_euclidiana(a, b):
    """Calcula a distância Euclidiana entre dois vetores."""
    diferenca = a - b
    soma_quadrados = np.sum(diferenca ** 2)
    return np.sqrt(soma_quadrados)


def distancia_manhattan(a, b):
    """Calcula a distância Manhattan entre dois vetores."""
    diferenca = np.abs(a - b)
    return np.sum(diferenca)


class KNNClassificador:
    """Implementação manual do kNN para classificação."""

    def __init__(self, k=3, distancia="euclidiana"):

        if (
            not isinstance(k, (int, np.integer))
            or isinstance(k, (bool, np.bool_))
            or k <= 0
        ):
            raise ValueError("k deve ser um inteiro positivo.")

        if distancia not in ["euclidiana", "manhattan"]:
            raise ValueError(
                "Distância deve ser 'euclidiana' ou 'manhattan'."
            )

        self.k = k
        self.distancia = distancia

        self.X_treino = None
        self.y_treino = None

    def treinar(self, X, y):
        """Armazena os dados de treinamento."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional.")

        if y.ndim != 1:
            raise ValueError("y deve ser um vetor.")

        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X e y devem ter o mesmo tamanho não vazio.")

        if self.k > len(X):
            raise ValueError(
                "k não pode ser maior que a quantidade de exemplos."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X contém valores numéricos inválidos.")

        self.X_treino = X.copy()
        self.y_treino = y.copy()

    def calcular_distancia(self, a, b):
        """Seleciona a função de distância."""

        if self.distancia == "euclidiana":
            return distancia_euclidiana(a, b)

        return distancia_manhattan(a, b)

    def prever_um(self, exemplo):
        """Classifica uma única instância."""

        if self.X_treino is None:
            raise ValueError(
                "O modelo precisa ser treinado antes da previsão."
            )

        exemplo = np.asarray(exemplo, dtype=float)

        if exemplo.shape != (self.X_treino.shape[1],):
            raise ValueError(
                "O exemplo possui quantidade incorreta de atributos."
            )

        distancias = []

        for registro in self.X_treino:
            distancia = self.calcular_distancia(exemplo, registro)
            distancias.append(distancia)

        distancias = np.array(distancias)

        indices_ordenados = np.argsort(
            distancias,
            kind="stable"
        )

        indices_vizinhos = indices_ordenados[:self.k]

        classes_vizinhos = self.y_treino[indices_vizinhos]

        classes, contagens = np.unique(
            classes_vizinhos,
            return_counts=True
        )

        indice_vencedor = np.argmax(contagens)

        return classes[indice_vencedor]

    def prever(self, X):
        """Classifica várias instâncias."""

        if self.X_treino is None:
            raise ValueError("O modelo ainda não foi treinado.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional.")

        if X.shape[1] != self.X_treino.shape[1]:
            raise ValueError("Quantidade incorreta de atributos.")

        previsoes = []

        for exemplo in X:
            classe_prevista = self.prever_um(exemplo)
            previsoes.append(classe_prevista)

        return np.array(previsoes)
