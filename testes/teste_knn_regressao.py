import time
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.normalizacao import Padronizador
from src.regressao.knn import KNNRegressor


def calcular_mae(y_real, y_previsto):
    """
    Calcula manualmente o erro absoluto médio.
    """

    erros = np.abs(y_real - y_previsto)

    return float(np.mean(erros))


def main():

    print("\n========== TESTE kNN REGRESSÃO ==========")

    # 1. Carrega o dataset.
    X, y, atributos = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )

    y = np.asarray(y, dtype=float)

    print(f"Instâncias: {len(X)}")
    print(f"Atributos preditores: {X.shape[1]}")

    # 2. Embaralha os dados.
    gerador = np.random.default_rng(42)

    indices = gerador.permutation(len(X))

    # 3. Divide em treinamento e teste.
    quantidade_treino = int(len(X) * 0.8)

    indices_treino = indices[:quantidade_treino]
    indices_teste = indices[quantidade_treino:]

    X_treino = X[indices_treino]
    y_treino = y[indices_treino]

    X_teste = X[indices_teste]
    y_teste = y[indices_teste]

    print(f"Treinamento: {len(X_treino)} instâncias")
    print(f"Teste: {len(X_teste)} instâncias")

    # 4. Padroniza os atributos.
    padronizador = Padronizador()

    X_treino = padronizador.treinar_transformar(
        X_treino
    )

    X_teste = padronizador.transformar(
        X_teste
    )

    print("\nPadronização realizada com sucesso.")

    # 5. Calcula o baseline.
    media_treinamento = np.mean(y_treino)

    previsoes_baseline = np.full(
        len(y_teste),
        media_treinamento
    )

    mae_baseline = calcular_mae(
        y_teste,
        previsoes_baseline
    )

    print("\n========== BASELINE ==========")

    print(
        f"Qualidade média do treinamento: "
        f"{media_treinamento:.4f}"
    )

    print(
        f"MAE baseline: {mae_baseline:.4f}"
    )

    # 6. Testa as duas distâncias.
    for distancia in ["euclidiana", "manhattan"]:

        print(
            f"\n========== kNN {distancia.upper()} =========="
        )

        modelo = KNNRegressor(
            k=5,
            distancia=distancia
        )

        inicio_treino = time.perf_counter()

        modelo.treinar(
            X_treino,
            y_treino
        )

        tempo_treino = (
            time.perf_counter() - inicio_treino
        )

        inicio_teste = time.perf_counter()

        previsoes = modelo.prever(
            X_teste
        )

        tempo_teste = (
            time.perf_counter() - inicio_teste
        )

        mae = calcular_mae(
            y_teste,
            previsoes
        )

        print(f"k: {modelo.k}")
        print(f"MAE: {mae:.4f}")

        print(
            f"Tempo de treinamento: "
            f"{tempo_treino:.6f} segundos"
        )

        print(
            f"Tempo de teste: "
            f"{tempo_teste:.6f} segundos"
        )

        print("\nPrimeiras 10 previsões:")

        for i in range(10):

            print(
                f"Real: {y_teste[i]:.1f} | "
                f"Previsto: {previsoes[i]:.2f}"
            )


if __name__ == "__main__":
    main()
