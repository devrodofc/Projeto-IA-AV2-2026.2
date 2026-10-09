import numpy as np


class Padronizador:
    """
    Padronização Z-score implementada com NumPy.

    Os parâmetros são calculados somente
    com os dados de treinamento.
    """

    def __init__(self):
        self.medias = None
        self.desvios = None

    def treinar(self, X):
        """
        Calcula a média e o desvio-padrão
        de cada atributo.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or len(X) == 0:
            raise ValueError(
                "X deve ser uma matriz bidimensional não vazia."
            )

        if X.shape[1] == 0 or not np.all(np.isfinite(X)):
            raise ValueError(
                "X deve conter atributos numéricos válidos."
            )

        self.medias = np.mean(X, axis=0)

        self.desvios = np.std(X, axis=0)

        # Evita divisão por zero em atributos constantes.
        self.desvios[self.desvios == 0] = 1.0

        return self

    def transformar(self, X):
        """
        Aplica a padronização utilizando
        os parâmetros do treinamento.
        """

        if self.medias is None:
            raise ValueError(
                "O padronizador precisa ser treinado."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X deve ser uma matriz bidimensional."
            )

        if X.shape[1] != len(self.medias):
            raise ValueError(
                "Quantidade incorreta de atributos."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contém valores inválidos."
            )

        return (X - self.medias) / self.desvios

    def treinar_transformar(self, X):
        """
        Treina e transforma os dados.
        """

        self.treinar(X)

        return self.transformar(X)
