import time
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.normalizacao import Padronizador
from src.regressao.regressao_linear import RegressaoLinearMultipla


def main():

    print("\n========== REGRESSÃO LINEAR MÚLTIPLA ==========")

    X, y, atributos = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )

    y = np.asarray(y, dtype=float)

    gerador = np.random.default_rng(42)
    indices = gerador.permutation(len(X))

    quantidade_treino = int(len(X) * 0.8)

    indices_treino = indices[:quantidade_treino]
    indices_teste = indices[quantidade_treino:]

    X_treino = X[indices_treino]
    y_treino = y[indices_treino]

    X_teste = X[indices_teste]
    y_teste = y[indices_teste]

    padronizador = Padronizador()

    X_treino = padronizador.treinar_transformar(
        X_treino
    )

    X_teste = padronizador.transformar(
        X_teste
    )

    modelo = RegressaoLinearMultipla()

    inicio = time.perf_counter()

    modelo.treinar(X_treino, y_treino)

    tempo_treinamento = time.perf_counter() - inicio

    inicio = time.perf_counter()

    previsoes = modelo.prever(X_teste)

    tempo_teste = time.perf_counter() - inicio

    mae = np.mean(np.abs(y_teste - previsoes))

    soma_erros_quadrados = np.sum(
        (y_teste - previsoes) ** 2
    )

    soma_total_quadrados = np.sum(
        (y_teste - np.mean(y_teste)) ** 2
    )

    r2 = 1 - soma_erros_quadrados / soma_total_quadrados

    n = len(y_teste)
    p = X_teste.shape[1]

    r2_ajustado = (
        1 - (1 - r2) * (n - 1) / (n - p - 1)
    )

    print(f"Treinamento: {len(X_treino)} instâncias")
    print(f"Teste: {len(X_teste)} instâncias")

    print("\n========== RESULTADOS ==========")

    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"R² ajustado: {r2_ajustado:.4f}")

    print(
        f"Tempo de treinamento: "
        f"{tempo_treinamento:.6f} segundos"
    )

    print(
        f"Tempo de teste: "
        f"{tempo_teste:.6f} segundos"
    )

    print("\nIntercepto:")
    print(f"{modelo.intercepto:.6f}")

    print("\nCoeficientes padronizados:")

    for atributo, coeficiente in zip(
        atributos,
        modelo.coeficientes
    ):
        print(f"{atributo}: {coeficiente:.6f}")

    print("\nPrimeiras 10 previsões:")

    for i in range(10):
        print(
            f"Real: {y_teste[i]:.1f} | "
            f"Previsto: {previsoes[i]:.4f}"
        )


if __name__ == "__main__":
    main()
