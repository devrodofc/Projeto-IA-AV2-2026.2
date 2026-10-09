import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import criar_folds


def verificar_folds(y, folds):

    todos_indices_teste = []

    for numero, (treino, teste) in enumerate(
        folds,
        start=1
    ):

        print(
            f"Fold {numero}: "
            f"{len(treino)} treinamento | "
            f"{len(teste)} teste"
        )

        # Treinamento e teste não podem
        # compartilhar instâncias.
        assert len(
            np.intersect1d(treino, teste)
        ) == 0

        todos_indices_teste.extend(
            teste.tolist()
        )

    # Cada instância deve aparecer uma vez no teste.
    assert len(todos_indices_teste) == len(y)

    assert len(
        np.unique(todos_indices_teste)
    ) == len(y)


def main():

    print("\n========== CLASSIFICAÇÃO ==========")

    X, y, _ = carregar_arff(
        "datasets/classificacao/spambase.arff"
    )

    folds = criar_folds(
        y,
        quantidade_folds=5,
        estratificado=True,
        semente=42
    )

    verificar_folds(y, folds)

    print("\nDistribuição das classes por fold:")

    for numero, (_, teste) in enumerate(
        folds,
        start=1
    ):

        classes, quantidades = np.unique(
            y[teste],
            return_counts=True
        )

        distribuicao = dict(
            zip(classes, quantidades)
        )

        print(
            f"Fold {numero}: {distribuicao}"
        )

    print("\n========== REGRESSÃO ==========")

    X, y, _ = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )

    folds = criar_folds(
        y,
        quantidade_folds=5,
        estratificado=False,
        semente=42
    )

    verificar_folds(y, folds)

    print("\nTodos os testes passaram!")


if __name__ == "__main__":
    main()
