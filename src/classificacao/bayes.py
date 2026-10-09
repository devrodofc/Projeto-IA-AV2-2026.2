import numpy as np


class BayesUnivariado:
    """
    Implementação manual do Gaussian Naive Bayes.

    Assume independência condicional entre os atributos
    e utiliza uma distribuição normal para cada atributo.
    """

    def __init__(self, regularizacao=1e-9):

        if regularizacao <= 0:
            raise ValueError(
                "A regularização deve ser positiva."
            )

        self.regularizacao = regularizacao

        self.classes = None
        self.priores = {}
        self.medias = {}
        self.variancias = {}

    def treinar(self, X, y):
        """
        Calcula as probabilidades iniciais,
        médias e variâncias de cada classe.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                "X deve ser uma matriz e y deve ser um vetor."
            )

        if len(X) == 0 or len(X) != len(y):
            raise ValueError(
                "X e y devem possuir o mesmo tamanho não vazio."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contém valores numéricos inválidos."
            )

        self.classes = np.unique(y)

        self.priores = {}
        self.medias = {}
        self.variancias = {}

        quantidade_total = len(y)

        for classe in self.classes:

            # Seleciona os registros pertencentes à classe.
            X_classe = X[y == classe]

            # Calcula a probabilidade inicial da classe.
            self.priores[classe] = (
                len(X_classe) / quantidade_total
            )

            # Calcula a média de cada atributo.
            self.medias[classe] = np.mean(
                X_classe,
                axis=0
            )

            # Calcula a variância de cada atributo.
            variancias = np.var(
                X_classe,
                axis=0
            )

            # Evita variâncias iguais a zero.
            self.variancias[classe] = (
                variancias + self.regularizacao
            )

        self.quantidade_atributos = X.shape[1]

    def calcular_log_gaussiana(self, exemplo, classe):
        """
        Calcula a soma dos logaritmos das densidades
        Gaussianas dos atributos de uma classe.
        """

        media = self.medias[classe]
        variancia = self.variancias[classe]

        # Logaritmo da densidade normal.
        log_densidade = (
            -0.5 * np.log(2 * np.pi * variancia)
            - ((exemplo - media) ** 2)
            / (2 * variancia)
        )

        return np.sum(log_densidade)

    def prever_um(self, exemplo):
        """
        Prevê a classe de uma única instância.
        """

        if self.classes is None:
            raise ValueError(
                "O modelo precisa ser treinado antes da previsão."
            )

        exemplo = np.asarray(exemplo, dtype=float)

        if exemplo.shape != (self.quantidade_atributos,):
            raise ValueError(
                "Quantidade incorreta de atributos."
            )

        if not np.all(np.isfinite(exemplo)):
            raise ValueError(
                "O exemplo contém valores inválidos."
            )

        pontuacoes = []

        for classe in self.classes:

            # Logaritmo da probabilidade inicial.
            log_prior = np.log(
                self.priores[classe]
            )

            # Soma das log-densidades dos atributos.
            log_verossimilhanca = (
                self.calcular_log_gaussiana(
                    exemplo,
                    classe
                )
            )

            # Pontuação proporcional à probabilidade posterior.
            log_posterior = (
                log_prior + log_verossimilhanca
            )

            pontuacoes.append(log_posterior)

        indice_vencedor = np.argmax(pontuacoes)

        return self.classes[indice_vencedor]

    def prever(self, X):
        """
        Prevê as classes de várias instâncias.
        """

        if self.classes is None:
            raise ValueError(
                "O modelo precisa ser treinado antes da previsão."
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

        previsoes = []

        for exemplo in X:

            previsoes.append(
                self.prever_um(exemplo)
            )

        return np.array(previsoes)
