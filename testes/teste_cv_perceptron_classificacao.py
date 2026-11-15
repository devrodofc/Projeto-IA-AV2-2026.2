import time
import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import k_fold
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_classificacao
from src.utils.exportacao import salvar_csv, resumir_resultados
from src.classificacao.perceptron import Perceptron


# Configuração escolhida na varredura de hiperparâmetros
# (scripts/varrer_redes_neurais.py): maior F1 macro médio.
LEARNING_RATE = 1.0
EPOCAS = 200

# Classe tratada como positiva na matriz de confusão: a minoritária.
CLASSE_POSITIVA = "class1"


def main():

    print(
        "\n========== PERCEPTRON SIMPLES ==========",
        flush=True
    )

    print(f"Taxa de aprendizado: {LEARNING_RATE}")
    print(f"Épocas: {EPOCAS}")

    X, y, _ = carregar_arff(
        "datasets/classificacao/autoUniv-au1-1000.arff"
    )

    folds = k_fold(X, y, k=5, random_state=42)

    resultados = []
    detalhes = []

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

        modelo = Perceptron(
            learning_rate=LEARNING_RATE,
            n_epochs=EPOCAS
        )

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
            "modelo": "perceptron",
            "fold": numero,
            "acuracia": metricas["acuracia"],
            "precisao_macro": metricas["precisao_macro"],
            "recall_macro": metricas["recall_macro"],
            "f1_macro": metricas["f1_macro"],
            "tempo_treinamento": tempo_treinamento,
            "tempo_teste": tempo_teste
        })

        # Matriz de confusão e métricas da classe positiva.
        # Linhas: classes reais. Colunas: classes previstas.
        matriz = metricas["matriz_confusao"]
        positiva = metricas["por_classe"][CLASSE_POSITIVA]

        detalhes.append({
            "modelo": "perceptron",
            "fold": numero,
            "vp": int(matriz[0, 0]),
            "fn": int(matriz[0, 1]),
            "fp": int(matriz[1, 0]),
            "vn": int(matriz[1, 1]),
            "sensibilidade": positiva["recall"],
            "especificidade": positiva["especificidade"],
            "precisao": positiva["precisao"],
            "f1": positiva["f1"]
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
            f"Matriz de confusão "
            f"({' / '.join(metricas['classes'])}):",
            flush=True
        )

        print(matriz, flush=True)

        print(
            f"Sensibilidade: {positiva['recall']:.4f}",
            flush=True
        )

        print(
            f"Especificidade: {positiva['especificidade']:.4f}",
            flush=True
        )

        print(
            f"Precisão: {positiva['precisao']:.4f}",
            flush=True
        )

        print(
            f"F1: {positiva['f1']:.4f}",
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

    for registros in (resultados, detalhes):

        for metrica in registros[0]:

            if metrica in ("modelo", "fold", "vp", "fn", "fp", "vn"):
                continue

            valores = np.array([
                registro[metrica]
                for registro in registros
            ])

            media = np.mean(valores)
            desvio = np.std(valores, ddof=1)

            print(
                f"{metrica}: "
                f"{media:.4f} ± {desvio:.4f}",
                flush=True
            )

    # Cada amostra é testada uma vez: a soma dos folds
    # é a matriz de confusão do dataset inteiro.
    print(
        f"\nMatriz de confusão somada nos 5 folds "
        f"(positiva: {CLASSE_POSITIVA}):"
    )

    for sigla in ("vp", "fn", "fp", "vn"):
        total = sum(detalhe[sigla] for detalhe in detalhes)
        print(f"{sigla.upper()}: {total}")

    caminho_folds = salvar_csv(
        "resultados/classificacao/perceptron_folds.csv",
        resultados
    )

    resumo = resumir_resultados(
        resultados,
        "perceptron"
    )

    caminho_resumo = salvar_csv(
        "resultados/classificacao/perceptron_resumo.csv",
        [resumo]
    )

    caminho_detalhes = salvar_csv(
        "resultados/classificacao/perceptron_matriz_confusao.csv",
        detalhes
    )

    print("\nArquivos exportados:")
    print(caminho_folds)
    print(caminho_resumo)
    print(caminho_detalhes)


if __name__ == "__main__":
    main()
