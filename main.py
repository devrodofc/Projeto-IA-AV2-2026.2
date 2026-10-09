import numpy as np

from src.utils.dados import carregar_arff


def main():

    caminho = "datasets/classificacao/autoUniv-au1-1000.arff"

    X, y, atributos = carregar_arff(
        caminho,
        atributo_alvo="Class"
    )

    print("\n=== ANÁLISE DO DATASET ===")

    print(f"Instâncias: {X.shape[0]}")
    print(f"Atributos preditores: {X.shape[1]}")

    print("\nNomes dos atributos:")
    print(atributos)

    print("\nPrimeiros 5 registros:")
    print(X[:5])

    print("\nPrimeiras 5 classes:")
    print(y[:5])

    print("\nDistribuição das classes:")

    classes, quantidades = np.unique(
        y,
        return_counts=True
    )

    for classe, quantidade in zip(classes, quantidades):

        percentual = quantidade / len(y) * 100

        print(
            f"{classe}: {quantidade} "
            f"({percentual:.2f}%)"
        )


if __name__ == "__main__":
    main()