import json
from pathlib import Path

import numpy as np

from src.utils.dados import carregar_arff
from src.utils.validacao import k_fold
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_classificacao, avaliar_regressao
from src.classificacao.bayes import NaiveBayes


VALORES_K = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]


def ordenar_vizinhos(X_treino, X_teste, distancia, tamanho_bloco=200):
    """
    Calcula as distâncias de cada exemplo de teste para todo o
    treino e devolve os índices ordenados (mesma regra da classe KNN).
    """

    ordens = []

    for inicio in range(0, len(X_teste), tamanho_bloco):
        bloco = X_teste[inicio:inicio + tamanho_bloco]
        diferencas = bloco[:, None, :] - X_treino[None, :, :]

        if distancia == "manhattan":
            distancias = np.sum(np.abs(diferencas), axis=2)
        else:
            distancias = np.sqrt(np.sum(diferencas ** 2, axis=2))

        ordens.append(np.argsort(distancias, axis=1))

    return np.vstack(ordens)


def media_desvio(valores):
    valores = np.asarray(valores, dtype=float)
    return [float(np.mean(valores)), float(np.std(valores, ddof=1))]


def resumir(registros):
    return {
        metrica: media_desvio([r[metrica] for r in registros])
        for metrica in registros[0]
    }


def preparar_folds(X, y):
    folds = []

    for X_treino, X_teste, y_treino, y_teste in k_fold(X, y, k=5, random_state=42):
        padronizador = Padronizador()
        X_treino = padronizador.treinar_transformar(X_treino)
        X_teste = padronizador.transformar(X_teste)
        folds.append((X_treino, X_teste, y_treino, y_teste))

    return folds


def varrer_classificacao():
    X, y, _ = carregar_arff("datasets/classificacao/autoUniv-au1-1000.arff")
    folds = preparar_folds(X, y)
    resultado = {"knn_euclidiana": {}, "knn_manhattan": {}, "bayes": {}}

    for distancia in ("euclidiana", "manhattan"):
        por_k = {k: [] for k in VALORES_K}

        for X_treino, X_teste, y_treino, y_teste in folds:
            ordem = ordenar_vizinhos(X_treino, X_teste, distancia)
            classes, y_codificado = np.unique(y_treino, return_inverse=True)

            for k in VALORES_K:
                rotulos = y_codificado[ordem[:, :k]]
                contagens = np.apply_along_axis(
                    np.bincount, 1, rotulos, minlength=len(classes)
                )
                previsoes = classes[np.argmax(contagens, axis=1)]
                m = avaliar_classificacao(y_teste, previsoes)
                por_k[k].append({
                    "acuracia": m["acuracia"],
                    "precisao_macro": m["precisao_macro"],
                    "recall_macro": m["recall_macro"],
                    "f1_macro": m["f1_macro"],
                })

        resultado[f"knn_{distancia}"] = {
            str(k): resumir(registros) for k, registros in por_k.items()
        }
        print(f"Classificação kNN {distancia}: ok", flush=True)

    # O Naive Bayes da aula não tem hiperparâmetro: uma única configuração.
    registros = []

    for X_treino, X_teste, y_treino, y_teste in folds:
        modelo = NaiveBayes()
        modelo.fit(X_treino, y_treino)
        m = avaliar_classificacao(y_teste, modelo.predict(X_teste))
        registros.append({
            "acuracia": m["acuracia"],
            "precisao_macro": m["precisao_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
        })

    resultado["bayes"] = resumir(registros)

    print("Classificação Bayes: ok", flush=True)

    classes, contagens = np.unique(y, return_counts=True)
    resultado["baseline_acuracia"] = float(np.max(contagens) / len(y))

    return resultado


def varrer_regressao():
    X, y, _ = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )
    folds = preparar_folds(X, y)
    resultado = {}

    for distancia in ("euclidiana", "manhattan"):
        por_k = {k: [] for k in VALORES_K}

        for X_treino, X_teste, y_treino, y_teste in folds:
            ordem = ordenar_vizinhos(X_treino, X_teste, distancia)

            for k in VALORES_K:
                previsoes = np.mean(y_treino[ordem[:, :k]], axis=1)
                m = avaliar_regressao(y_teste, previsoes, X.shape[1])
                por_k[k].append({
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "r2_ajustado": m["r2_ajustado"],
                })

        resultado[f"knn_{distancia}"] = {
            str(k): resumir(registros) for k, registros in por_k.items()
        }
        print(f"Regressão kNN {distancia}: ok", flush=True)

    return resultado


def main():
    dados = {
        "valores_k": VALORES_K,
        "classificacao": varrer_classificacao(),
        "regressao": varrer_regressao(),
    }

    caminho = Path("resultados/hiperparametros.json")
    caminho.write_text(json.dumps(dados, indent=2), encoding="utf-8")
    print(f"Arquivo gerado: {caminho}")


if __name__ == "__main__":
    main()
