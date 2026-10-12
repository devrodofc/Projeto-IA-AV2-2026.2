import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import k_fold


def verificar_folds(n, k=5):
    """
    Usa os próprios índices como X e y para verificar
    se os folds de teste são disjuntos e cobrem
    todas as instâncias.
    """

    indices = np.arange(n)

    folds = k_fold(indices, indices, k=k, random_state=42)

    assert len(folds) == k

    todos_indices_teste = []

    for numero, (X_treino, X_teste, y_treino, y_teste) in enumerate(
        folds,
        start=1
    ):

        print(
            f"Fold {numero}: "
            f"{len(X_treino)} treinamento | "
            f"{len(X_teste)} teste"
        )

        # Treinamento e teste não podem
        # compartilhar instâncias.
        assert len(
            np.intersect1d(X_treino, X_teste)
        ) == 0

        assert len(X_treino) + len(X_teste) == n

        todos_indices_teste.extend(
            X_teste.tolist()
        )

    # Cada instância deve aparecer uma vez no teste.
    assert len(todos_indices_teste) == n

    assert len(
        np.unique(todos_indices_teste)
    ) == n


def main():

    print("\n========== CLASSIFICAÇÃO ==========")

    X, y, _ = carregar_arff(
        "datasets/classificacao/autoUniv-au1-1000.arff"
    )

    verificar_folds(len(X))

    print("\nDistribuição das classes por fold:")

    folds = k_fold(X, y, k=5, random_state=42)

    for numero, (_, _, _, y_teste) in enumerate(
        folds,
        start=1
    ):

        classes, quantidades = np.unique(
            y_teste,
            return_counts=True
        )

        distribuicao = {
            str(classe): int(quantidade)
            for classe, quantidade in zip(classes, quantidades)
        }

        print(
            f"Fold {numero}: {distribuicao}"
        )

    print("\n========== REGRESSÃO ==========")

    X, y, _ = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )

    verificar_folds(len(X))

    print("\nTodos os testes passaram!")


if __name__ == "__main__":
    main()
