import sys
import time
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import criar_folds
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_classificacao
from src.classificacao.bayes import (
    BayesUnivariado,
    BayesMultivariado
)


def main():

    if len(sys.argv) != 2:
        raise ValueError(
            "Informe 'univariado' ou 'multivariado'."
        )

    tipo = sys.argv[1].lower()

    if tipo not in ("univariado", "multivariado"):
        raise ValueError(
            "Use 'univariado' ou 'multivariado'."
        )

    print(
        f"\n========== BAYES {tipo.upper()} ==========",
        flush=True
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

        print(
            f"\n========== FOLD {numero} ==========",
            flush=True
        )

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]

        X_teste = X[indices_teste]
        y_teste = y[indices_teste]

        # Ajusta a padronização apenas no treino.
        padronizador = Padronizador()

        X_treino = padronizador.treinar_transformar(
            X_treino
        )

        X_teste = padronizador.transformar(
            X_teste
        )

        if tipo == "univariado":
            modelo = BayesUnivariado(
                regularizacao=1e-9
            )
        else:
            modelo = BayesMultivariado(
                regularizacao=1e-6
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
        desvio = np.std(valores, ddof=1)

        print(
            f"{metrica}: "
            f"{media:.4f} ± {desvio:.4f}",
            flush=True
        )


if __name__ == "__main__":
    main()
