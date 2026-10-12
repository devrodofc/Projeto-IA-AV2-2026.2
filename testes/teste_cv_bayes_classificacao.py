import time
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import k_fold
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_classificacao
from src.utils.exportacao import salvar_csv, resumir_resultados
from src.classificacao.bayes import NaiveBayes


def main():

    tipo = "univariado"

    print(
        f"\n========== BAYES {tipo.upper()} ==========",
        flush=True
    )

    X, y, _ = carregar_arff(
        "datasets/classificacao/autoUniv-au1-1000.arff"
    )

    folds = k_fold(X, y, k=5, random_state=42)

    resultados = []

    for numero, (X_treino, X_teste, y_treino, y_teste) in enumerate(
        folds,
        start=1
    ):

        print(
            f"\n========== FOLD {numero} ==========",
            flush=True
        )

        # Ajusta a padronização apenas no treino.
        padronizador = Padronizador()

        X_treino = padronizador.treinar_transformar(
            X_treino
        )

        X_teste = padronizador.transformar(
            X_teste
        )

        modelo = NaiveBayes()

        inicio = time.perf_counter()

        modelo.fit(X_treino, y_treino)

        tempo_treinamento = (
            time.perf_counter() - inicio
        )

        inicio = time.perf_counter()

        previsoes = modelo.predict(X_teste)

        tempo_teste = (
            time.perf_counter() - inicio
        )

        metricas = avaliar_classificacao(
            y_teste,
            previsoes
        )

        resultados.append({
            "modelo": f"bayes_{tipo}",
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

    print(
        "\n========== RESULTADOS FINAIS ==========",
        flush=True
    )

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
            f"{media:.4f} ± {desvio:.4f}",
            flush=True
        )


    caminho_folds = salvar_csv(
        f"resultados/classificacao/bayes_{tipo}_folds.csv",
        resultados
    )

    resumo = resumir_resultados(
        resultados,
        f"bayes_{tipo}"
    )

    caminho_resumo = salvar_csv(
        f"resultados/classificacao/bayes_{tipo}_resumo.csv",
        [resumo]
    )

    print("\nArquivos exportados:")
    print(caminho_folds)
    print(caminho_resumo)


if __name__ == "__main__":
    main()
