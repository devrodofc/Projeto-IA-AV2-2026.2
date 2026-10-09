import numpy as np


def criar_folds(
    y,
    quantidade_folds=5,
    estratificado=False,
    semente=42
):
    """
    Cria folds para validação cruzada.

    Parâmetros:
        y: vetor de valores reais.
        quantidade_folds: quantidade de partições.
        estratificado: preserva a proporção das classes.
        semente: controla a aleatoriedade.

    Retorna:
        Lista de tuplas:
        (indices_treinamento, indices_teste)
    """

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y deve ser um vetor.")

    n = len(y)

    if (
        not isinstance(quantidade_folds, (int, np.integer))
        or isinstance(quantidade_folds, (bool, np.bool_))
        or quantidade_folds < 2
        or quantidade_folds > n
    ):
        raise ValueError(
            "A quantidade de folds deve estar entre 2 e n."
        )

    gerador = np.random.default_rng(semente)

    folds = [
        [] for _ in range(quantidade_folds)
    ]

    if estratificado:

        classes = np.unique(y)

        for classe in classes:

            indices_classe = np.where(
                y == classe
            )[0]

            gerador.shuffle(indices_classe)

            # Distribui as instâncias da classe
            # entre os folds.
            for posicao, indice in enumerate(indices_classe):

                indice_fold = (
                    posicao % quantidade_folds
                )

                folds[indice_fold].append(
                    int(indice)
                )

    else:

        indices = gerador.permutation(n)

        partes = np.array_split(
            indices,
            quantidade_folds
        )

        folds = [
            parte.tolist()
            for parte in partes
        ]

    resultado = []

    todos_indices = np.arange(n)

    for indices_teste in folds:

        indices_teste = np.asarray(
            indices_teste,
            dtype=int
        )

        # Ordena os índices para manter
        # uma ordem consistente.
        indices_teste = np.sort(indices_teste)

        mascara_treinamento = np.ones(
            n,
            dtype=bool
        )

        mascara_treinamento[
            indices_teste
        ] = False

        indices_treinamento = todos_indices[
            mascara_treinamento
        ]

        resultado.append(
            (
                indices_treinamento,
                indices_teste
            )
        )

    return resultado
