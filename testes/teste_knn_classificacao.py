import numpy as np

from src.utils.dados import carregar_arff
from src.classificacao.knn import KNNClassificador


def main():

    print("\n========== TESTE DO kNN ==========")

    # 1. Carrega o dataset.
    X, y, _ = carregar_arff(
        "datasets/classificacao/autoUniv-au1-1000.arff",
        atributo_alvo="Class"
    )

    # 2. Embaralha os índices de maneira reproduzível.
    gerador = np.random.default_rng(42)

    indices = gerador.permutation(len(X))

    # 3. Separa 80% para treinamento e 20% para teste.
    quantidade_treino = int(len(X) * 0.8)

    indices_treino = indices[:quantidade_treino]
    indices_teste = indices[quantidade_treino:]

    X_treino = X[indices_treino]
    y_treino = y[indices_treino]

    X_teste = X[indices_teste]
    y_teste = y[indices_teste]

    print(f"Treinamento: {len(X_treino)} instâncias")
    print(f"Teste: {len(X_teste)} instâncias")

    # 4. Testa as duas distâncias.
    for distancia in ["euclidiana", "manhattan"]:

        print(f"\n--- kNN {distancia.upper()} ---")

        modelo = KNNClassificador(
            k=3,
            distancia=distancia
        )

        modelo.treinar(X_treino, y_treino)

        previsoes = modelo.prever(X_teste)

        # Cálculo manual da acurácia.
        acertos = np.sum(previsoes == y_teste)

        acuracia = acertos / len(y_teste)

        print(f"Acertos: {acertos}")
        print(f"Total de testes: {len(y_teste)}")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Acurácia percentual: {acuracia * 100:.2f}%")

        print("\nPrimeiras 10 previsões:")

        for i in range(10):

            print(
                f"Real: {y_teste[i]} | "
                f"Previsto: {previsoes[i]}"
            )


if __name__ == "__main__":
    main()