import numpy as np

from src.utils.dados import carregar_arff


def analisar_classificacao():
    """
    Apresenta informações do dataset Spambase,
    utilizado nos experimentos de classificação.
    """

    print("\n========== CLASSIFICAÇÃO ==========")

    caminho = "datasets/classificacao/spambase.arff"

    X, y, atributos = carregar_arff(caminho)

    print("Dataset: Spambase")
    print("Fonte: OpenML - ID 44")
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
            f"Classe {classe}: {quantidade} "
            f"({percentual:.2f}%)"
        )


def analisar_regressao():
    """
    Apresenta informações do dataset Wine Quality,
    utilizado nos experimentos de regressão.
    """

    print("\n========== REGRESSÃO ==========")

    caminho = "datasets/regressao/wine_quality.arff"

    X, y, atributos = carregar_arff(
        caminho,
        atributo_alvo="quality"
    )

    print("Dataset: Wine Quality")
    print("Fonte: OpenML - ID 287")
    print(f"Instâncias: {X.shape[0]}")
    print(f"Atributos preditores: {X.shape[1]}")

    print("\nEstatísticas da variável-alvo:")

    print(f"Menor qualidade: {np.min(y)}")
    print(f"Maior qualidade: {np.max(y)}")
    print(f"Qualidade média: {np.mean(y):.2f}")
    print(f"Desvio-padrão: {np.std(y):.2f}")

    print("\nPrimeiros cinco valores de qualidade:")
    print(y[:5])


def main():

    print("=" * 55)
    print("PROJETO DE INTELIGÊNCIA ARTIFICIAL - AV2")
    print("Comparação de algoritmos de classificação e regressão")
    print("=" * 55)

    analisar_classificacao()

    analisar_regressao()

    print("\nAnálise dos datasets concluída.")
    print(
        "Consulte o README.md para executar "
        "os experimentos."
    )


if __name__ == "__main__":
    main()
