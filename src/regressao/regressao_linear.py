import numpy as np


class RegressaoLinearMultipla:
    """
    Regressão Linear Múltipla utilizando
    Mínimos Quadrados Ordinários (OLS).
    """

    def __init__(self):
        self.coeficientes = None
        self.intercepto = None
        self.quantidade_atributos = None

    def treinar(self, X, y):
        """
        Estima os coeficientes pelo método OLS.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                "X deve ser uma matriz e y deve ser um vetor."
            )

        if len(X) == 0 or len(X) != len(y):
            raise ValueError(
                "X e y devem possuir o mesmo tamanho não vazio."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X deve possuir pelo menos um atributo."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X contém valores inválidos.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contém valores inválidos.")

        self.quantidade_atributos = X.shape[1]

        # Adiciona uma coluna de 1 para o intercepto.
        coluna_uns = np.ones((len(X), 1))

        X_com_intercepto = np.column_stack(
            (coluna_uns, X)
        )

        # Resolve o problema de mínimos quadrados.
        parametros, _, _, _ = np.linalg.lstsq(
            X_com_intercepto,
            y,
            rcond=None
        )

        self.intercepto = float(parametros[0])
        self.coeficientes = parametros[1:].copy()

    def prever(self, X):
        """
        Realiza previsões utilizando os coeficientes.
        """

        if self.coeficientes is None:
            raise ValueError(
                "O modelo precisa ser treinado."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X deve ser uma matriz bidimensional."
            )

        if X.shape[1] != self.quantidade_atributos:
            raise ValueError(
                "Quantidade incorreta de atributos."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X contém valores inválidos.")

        return (
            self.intercepto
            + X @ self.coeficientes
        )
