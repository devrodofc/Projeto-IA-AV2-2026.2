import time
import sys
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import criar_folds
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_classificacao
from src.utils.exportacao import salvar_csv, resumir_resultados
from src.classificacao.knn import KNNClassificador


def main():

    distancia = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "euclidiana"
    )

    if distancia not in ("euclidiana", "manhattan"):
        raise ValueError(
            "Use 'euclidiana' ou 'manhattan'."
        )

    print(
        f"\n========== VALIDAÇÃO CRUZADA kNN "
        f"{distancia.upper()} =========="
    )

    X, y, _ = carregar_arff(
        "datasets/classificacao/spambase.arff"
    )

    folds = criar_folds(
        y,
        quantidade_folds=5,
        estratificado=True,
        semente=42
    )

    resultados = []

    for numero, (indices_treino, indices_teste) in enumerate(
        folds,
        start=1
    ):

        print(f"\n========== FOLD {numero} ==========", flush=True)

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

        modelo = KNNClassificador(
            k=3,
            distancia=distancia
        )

        inicio = time.perf_counter()

        modelo.treinar(X_treino, y_treino)

        tempo_treinamento = (
            time.perf_counter() - inicio
        )

        inicio = time.perf_counter()

        previsoes = modelo.prever(X_teste)

        tempo_teste = (
            time.perf_counter() - inicio
        )

        metricas = avaliar_classificacao(
            y_teste,
            previsoes
        )

        resultados.append({
            "modelo": f"knn_{distancia}",
            "fold": numero,
            "acuracia": metricas["acuracia"],
            "precisao_macro": metricas["precisao_macro"],
            "recall_macro": metricas["recall_macro"],
            "f1_macro": metricas["f1_macro"],
            "tempo_treinamento": tempo_treinamento,
            "tempo_teste": tempo_teste
        })

        print(
            f"Acurácia: {metricas['acuracia']:.4f}",
            flush=True
        )

        print(
            f"F1 macro: {metricas['f1_macro']:.4f}",
            flush=True
        )

        print(
            f"Treinamento: {tempo_treinamento:.4f}s",
            flush=True
        )

        print(
            f"Teste: {tempo_teste:.4f}s",
            flush=True
        )

    print("\n========== RESULTADOS FINAIS ==========")

    for metrica in resultados[0]:

        if metrica in ("modelo", "fold"):
            continue

        valores = np.array([
            resultado[metrica]
            for resultado in resultados
        ])

        media = np.mean(valores)
        desvio = np.std(valores, ddof=1)

        print(
            f"{metrica}: "
            f"{media:.4f} ± {desvio:.4f}"
        )


    # Exporta os resultados individuais dos cinco folds.
    caminho_folds = salvar_csv(
        f"resultados/classificacao/knn_{distancia}_folds.csv",
        resultados
    )

    # Calcula e exporta as médias e desvios.
    resumo = resumir_resultados(
        resultados,
        f"knn_{distancia}"
    )

    caminho_resumo = salvar_csv(
        f"resultados/classificacao/knn_{distancia}_resumo.csv",
        [resumo]
    )

    print("\nArquivos exportados:")
    print(caminho_folds)
    print(caminho_resumo)


if __name__ == "__main__":
    main()
