import numpy as np


def distancia_euclidiana(a, b):
    """
    Calcula a distância Euclidiana.
    """

    diferenca = a - b

    return np.sqrt(
        np.sum(diferenca ** 2)
    )


def distancia_manhattan(a, b):
    """
    Calcula a distância Manhattan.
    """

    diferenca = np.abs(a - b)

    return np.sum(diferenca)


class KNNRegressor:
    """
    Implementação manual do kNN para regressão.

    A previsão corresponde à média dos valores
    dos k vizinhos mais próximos.
    """

    def __init__(self, k=3, distancia="euclidiana"):

        if (
            not isinstance(k, (int, np.integer))
            or isinstance(k, (bool, np.bool_))
            or k <= 0
        ):
            raise ValueError(
                "k deve ser um inteiro positivo."
            )

        if distancia not in ["euclidiana", "manhattan"]:
            raise ValueError(
                "Distância deve ser 'euclidiana' ou 'manhattan'."
            )

        self.k = k
        self.distancia = distancia

        self.X_treino = None
        self.y_treino = None

    def treinar(self, X, y):
        """
        Armazena os exemplos de treinamento.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X deve ser uma matriz bidimensional."
            )

        if y.ndim != 1:
            raise ValueError(
                "y deve ser um vetor."
            )

        if len(X) == 0 or len(X) != len(y):
            raise ValueError(
                "X e y devem possuir o mesmo tamanho não vazio."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X deve possuir pelo menos um atributo."
            )

        if self.k > len(X):
            raise ValueError(
                "k não pode ser maior que a quantidade de exemplos."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contém valores numéricos inválidos."
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "y contém valores numéricos inválidos."
            )

        self.X_treino = X.copy()
        self.y_treino = y.copy()

    def calcular_distancia(self, a, b):
        """
        Calcula a distância selecionada.
        """

        if self.distancia == "euclidiana":
            return distancia_euclidiana(a, b)

        return distancia_manhattan(a, b)

    def prever_um(self, exemplo):
        """
        Prevê o valor de uma única instância.
        """

        if self.X_treino is None:
            raise ValueError(
                "O modelo precisa ser treinado antes da previsão."
            )

        exemplo = np.asarray(exemplo, dtype=float)

        if exemplo.shape != (self.X_treino.shape[1],):
            raise ValueError(
                "Quantidade incorreta de atributos."
            )

        if not np.all(np.isfinite(exemplo)):
            raise ValueError(
                "O exemplo contém valores inválidos."
            )

        distancias = []

        # Calcula a distância até cada exemplo conhecido.
        for registro in self.X_treino:

            distancia = self.calcular_distancia(
                exemplo,
                registro
            )

            distancias.append(distancia)

        distancias = np.asarray(distancias)

        # Ordena os índices pelas distâncias.
        indices_ordenados = np.argsort(
            distancias,
            kind="stable"
        )

        # Seleciona os k vizinhos mais próximos.
        indices_vizinhos = indices_ordenados[:self.k]

        # Obtém os valores reais desses vizinhos.
        valores_vizinhos = self.y_treino[
            indices_vizinhos
        ]

        # Calcula a média dos valores.
        previsao = np.mean(valores_vizinhos)

        return float(previsao)

    def prever(self, X):
        """
        Prevê os valores de várias instâncias.
        """

        if self.X_treino is None:
            raise ValueError(
                "O modelo precisa ser treinado antes da previsão."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X deve ser uma matriz bidimensional."
            )

        if X.shape[1] != self.X_treino.shape[1]:
            raise ValueError(
                "Quantidade incorreta de atributos."
            )

        previsoes = []

        for exemplo in X:

            previsao = self.prever_um(exemplo)

            previsoes.append(previsao)

        return np.asarray(previsoes, dtype=float)
