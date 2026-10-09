import sys
import time
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import criar_folds
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_regressao

from src.regressao.knn import KNNRegressor
from src.regressao.regressao_linear import RegressaoLinearMultipla


def criar_modelo(tipo):
    """
    Cria o modelo de regressão solicitado.
    """

    if tipo == "knn_euclidiana":
        return KNNRegressor(
            k=5,
            distancia="euclidiana"
        )

    if tipo == "knn_manhattan":
        return KNNRegressor(
            k=5,
            distancia="manhattan"
        )

    if tipo == "linear":
        return RegressaoLinearMultipla()

    raise ValueError(
        "Modelo inválido. Utilize: "
        "knn_euclidiana, knn_manhattan ou linear."
    )


def main():

    if len(sys.argv) != 2:
        raise ValueError(
            "Informe o modelo: "
            "knn_euclidiana, knn_manhattan ou linear."
        )

    tipo = sys.argv[1].lower()

    # Valida o argumento antes de carregar os dados.
    criar_modelo(tipo)

    print(
        f"\n========== REGRESSÃO: {tipo.upper()} ==========",
        flush=True
    )

    X, y, atributos = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )

    quantidade_atributos = X.shape[1]

    print(f"Instâncias: {len(X)}")
    print(f"Atributos preditores: {quantidade_atributos}")

    folds = criar_folds(
        y,
        quantidade_folds=5,
        estratificado=False,
        semente=42
    )

    resultados = []

    for numero, (indices_treino, indices_teste) in enumerate(
        folds,
        start=1
    ):

        print(
            f"\n========== FOLD {numero} ==========",
            flush=True
        )

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]

        X_teste = X[indices_teste]
        y_teste = y[indices_teste]

        # Evita vazamento de dados:
        # o padronizador aprende apenas com o treino.
        padronizador = Padronizador()

        X_treino = padronizador.treinar_transformar(
            X_treino
        )

        X_teste = padronizador.transformar(
            X_teste
        )

        # Cria um novo modelo para cada fold.
        modelo = criar_modelo(tipo)

        # Mede apenas o treinamento do modelo.
        inicio = time.perf_counter()

        modelo.treinar(X_treino, y_treino)

        tempo_treinamento = (
            time.perf_counter() - inicio
        )

        # Mede apenas a previsão.
        inicio = time.perf_counter()

        previsoes = modelo.prever(X_teste)

        tempo_teste = (
            time.perf_counter() - inicio
        )

        metricas = avaliar_regressao(
            y_teste,
            previsoes,
            quantidade_atributos
        )

        resultados.append({
            "mae": metricas["mae"],
            "r2": metricas["r2"],
            "r2_ajustado": metricas["r2_ajustado"],
            "tempo_treinamento": tempo_treinamento,
            "tempo_teste": tempo_teste
        })

        print(
            f"MAE: {metricas['mae']:.4f}",
            flush=True
        )

        print(
            f"R²: {metricas['r2']:.4f}",
            flush=True
        )

        print(
            f"R² ajustado: "
            f"{metricas['r2_ajustado']:.4f}",
            flush=True
        )

        print(
            f"Treinamento: {tempo_treinamento:.6f}s",
            flush=True
        )

        print(
            f"Teste: {tempo_teste:.4f}s",
            flush=True
        )

    print(
        "\n========== RESULTADOS FINAIS ==========",
        flush=True
    )

    for metrica in resultados[0]:

        valores = np.array([
            resultado[metrica]
            for resultado in resultados
        ])

        media = np.mean(valores)

        # Desvio-padrão amostral.
        desvio = np.std(
            valores,
            ddof=1
        )

        print(
            f"{metrica}: "
            f"{media:.4f} ± {desvio:.4f}",
            flush=True
        )

    print(
        "\nValidação cruzada concluída!",
        flush=True
    )


if __name__ == "__main__":
    main()
