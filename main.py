import numpy as np

from src.utils.dados import carregar_arff


def analisar_classificacao():

    print("\n========== CLASSIFICAÇÃO ==========")

    caminho = "datasets/classificacao/autoUniv-au1-1000.arff"

    X, y, atributos = carregar_arff(
        caminho,
        atributo_alvo="Class"
    )

    print(f"Instâncias: {X.shape[0]}")
    print(f"Atributos preditores: {X.shape[1]}")

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


def analisar_regressao():

    print("\n========== REGRESSÃO ==========")

    caminho = "datasets/regressao/wine_quality.arff"

    X, y, atributos = carregar_arff(
        caminho,
        atributo_alvo="quality"
    )

    print(f"Instâncias: {X.shape[0]}")
    print(f"Atributos preditores: {X.shape[1]}")

    print("\nAtributos:")
    for atributo in atributos:
        print(f"- {atributo}")

    print("\nEstatísticas da variável-alvo:")

    print(f"Menor qualidade: {np.min(y)}")
    print(f"Maior qualidade: {np.max(y)}")
    print(f"Qualidade média: {np.mean(y):.2f}")
    print(f"Desvio padrão: {np.std(y):.2f}")

    print("\nPrimeiros 5 valores de qualidade:")
    print(y[:5])


def main():

    analisar_classificacao()

    analisar_regressao()


if __name__ == "__main__":
    main()