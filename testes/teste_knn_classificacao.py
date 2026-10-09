import numpy as np

from src.utils.dados import carregar_arff
from src.utils.metricas import avaliar_classificacao
from src.classificacao.knn import KNNClassificador


def main():

    print("\n========== TESTE DO kNN ==========")

    # 1. Carrega o dataset.
    X, y, _ = carregar_arff(
        "datasets/classificacao/autoUniv-au1-1000.arff",
        atributo_alvo="Class"
    )

    # 2. Embaralha os índices.
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

    # 4. Baseline: prever sempre a classe mais frequente
    # encontrada APENAS no conjunto de treinamento.
    classes_treino, contagens = np.unique(
        y_treino,
        return_counts=True
    )

    classe_majoritaria = classes_treino[
        np.argmax(contagens)
    ]

    previsoes_baseline = np.full(
        len(y_teste),
        classe_majoritaria
    )

    resultado_baseline = avaliar_classificacao(
        y_teste,
        previsoes_baseline
    )

    print("\n========== BASELINE ==========")
    print(f"Classe majoritária: {classe_majoritaria}")
    print(
        f"Acurácia: "
        f"{resultado_baseline['acuracia'] * 100:.2f}%"
    )
    print(
        f"F1 macro: "
        f"{resultado_baseline['f1_macro']:.4f}"
    )

    # 5. Avalia o kNN com as duas distâncias.
    for distancia in ["euclidiana", "manhattan"]:

        print(f"\n========== kNN {distancia.upper()} ==========")

        modelo = KNNClassificador(
            k=3,
            distancia=distancia
        )

        modelo.treinar(X_treino, y_treino)

        previsoes = modelo.prever(X_teste)

        resultado = avaliar_classificacao(
            y_teste,
            previsoes
        )

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

        print("\nPrimeiras 10 previsões:")

        for i in range(10):
            print(
                f"Real: {y_teste[i]} | "
                f"Previsto: {previsoes[i]}"
            )


if __name__ == "__main__":
    main()
