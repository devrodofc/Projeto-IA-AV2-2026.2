import numpy as np

from src.utils.dados import carregar_arff
from src.utils.metricas import avaliar_classificacao
from src.classificacao.bayes import BayesMultivariado


def main():

    print("\n========== BAYES MULTIVARIADO ==========")

    # 1. Carrega o dataset.
    X, y, _ = carregar_arff(
        "datasets/classificacao/autoUniv-au1-1000.arff",
        atributo_alvo="Class"
    )

    # 2. Utiliza a mesma semente dos testes anteriores.
    gerador = np.random.default_rng(42)
    indices = gerador.permutation(len(X))

    # 3. Divide em treinamento (80%) e teste (20%).
    quantidade_treino = int(len(X) * 0.8)

    indices_treino = indices[:quantidade_treino]
    indices_teste = indices[quantidade_treino:]

    X_treino = X[indices_treino]
    y_treino = y[indices_treino]

    X_teste = X[indices_teste]
    y_teste = y[indices_teste]

    print(f"Treinamento: {len(X_treino)} instâncias")
    print(f"Teste: {len(X_teste)} instâncias")

    # 4. Cria e treina o modelo.
    modelo = BayesMultivariado(
        regularizacao=1e-6
    )

    modelo.treinar(X_treino, y_treino)

    # 5. Realiza as previsões.
    previsoes = modelo.prever(X_teste)

    # 6. Calcula as métricas.
    resultado = avaliar_classificacao(
        y_teste,
        previsoes
    )

    print("\n========== RESULTADOS ==========")

    print(
        f"Acurácia: "
        f"{resultado['acuracia'] * 100:.2f}%"
    )

    print(
        f"Precisão macro: "
        f"{resultado['precisao_macro']:.4f}"
    )

    print(
        f"Recall macro: "
        f"{resultado['recall_macro']:.4f}"
    )

    print(
        f"F1 macro: "
        f"{resultado['f1_macro']:.4f}"
    )

    print("\nMétricas por classe:")

    for classe, metricas in resultado["por_classe"].items():

        print(f"\nClasse: {classe}")

        print(
            f"  Precisão: "
            f"{metricas['precisao']:.4f}"
        )

        print(
            f"  Recall: "
            f"{metricas['recall']:.4f}"
        )

        print(
            f"  F1-score: "
            f"{metricas['f1']:.4f}"
        )

    print("\nMatriz de confusão:")
    print("Ordem das classes:", resultado["classes"])
    print(resultado["matriz_confusao"])

    print("\nProbabilidades iniciais das classes:")

    for classe, probabilidade in modelo.priores.items():

        print(
            f"{classe}: {probabilidade * 100:.2f}%"
        )

    print("\nDimensões das matrizes de covariância:")

    for classe, matriz in modelo.covariancias.items():

        print(
            f"{classe}: {matriz.shape}"
        )

    print("\nPrimeiras 10 previsões:")

    for i in range(10):

        print(
            f"Real: {y_teste[i]} | "
            f"Previsto: {previsoes[i]}"
        )


if __name__ == "__main__":
    main()
