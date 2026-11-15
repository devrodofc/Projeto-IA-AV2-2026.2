import numpy as np

from scripts.varrer_hiperparametros import ordenar_vizinhos
from src.utils.dados import carregar_arff
from src.utils.validacao import k_fold
from src.utils.normalizacao import Padronizador
from src.utils.metricas import avaliar_classificacao, avaliar_regressao
from src.utils.exportacao import salvar_csv, resumir_resultados
from src.classificacao.bayes import NaiveBayes
from src.classificacao.perceptron import Perceptron
from src.regressao.regressao_linear import MRegression
from src.regressao.mlp import MLP
from testes import teste_cv_perceptron_classificacao as perceptron
from testes import teste_cv_regressao as regressao


# Mesmos valores de k dos experimentos principais.
K_CLASSIFICACAO = 3
K_REGRESSAO = 5


def preparar_folds(X, y, padronizar):
    """
    Mesmos cinco folds dos experimentos, com ou sem Z-score.
    """

    folds = []

    for X_treino, X_teste, y_treino, y_teste in k_fold(X, y, k=5, random_state=42):
        if padronizar:
            padronizador = Padronizador()
            X_treino = padronizador.treinar_transformar(X_treino)
            X_teste = padronizador.transformar(X_teste)
        folds.append((X_treino, X_teste, y_treino, y_teste))

    return folds


def prever_knn(X_treino, X_teste, y_treino, distancia, k, tarefa):
    """
    Mesma regra da classe KNN, com as distâncias calculadas em bloco.
    """

    ordem = ordenar_vizinhos(X_treino, X_teste, distancia)[:, :k]

    if tarefa == "regressao":
        return np.mean(y_treino[ordem], axis=1)

    classes, y_codificado = np.unique(y_treino, return_inverse=True)
    contagens = np.apply_along_axis(
        np.bincount, 1, y_codificado[ordem], minlength=len(classes)
    )
    return classes[np.argmax(contagens, axis=1)]


def prever(modelo, X_treino, X_teste, y_treino):
    modelo.fit(X_treino, y_treino)
    return np.asarray(modelo.predict(X_teste))


MODELOS_CLASSIFICACAO = {
    "knn_euclidiana": lambda Xa, Xb, ya: prever_knn(
        Xa, Xb, ya, "euclidiana", K_CLASSIFICACAO, "classificacao"
    ),
    "knn_manhattan": lambda Xa, Xb, ya: prever_knn(
        Xa, Xb, ya, "manhattan", K_CLASSIFICACAO, "classificacao"
    ),
    "bayes_univariado": lambda Xa, Xb, ya: prever(NaiveBayes(), Xa, Xb, ya),
    "perceptron": lambda Xa, Xb, ya: prever(
        Perceptron(
            learning_rate=perceptron.LEARNING_RATE,
            n_epochs=perceptron.EPOCAS
        ),
        Xa, Xb, ya
    ),
}

MODELOS_REGRESSAO = {
    "linear": lambda Xa, Xb, ya: prever(MRegression(), Xa, Xb, ya),
    "knn_euclidiana": lambda Xa, Xb, ya: prever_knn(
        Xa, Xb, ya, "euclidiana", K_REGRESSAO, "regressao"
    ),
    "knn_manhattan": lambda Xa, Xb, ya: prever_knn(
        Xa, Xb, ya, "manhattan", K_REGRESSAO, "regressao"
    ),
    "mlp": lambda Xa, Xb, ya: prever(regressao.criar_modelo("mlp"), Xa, Xb, ya),
}


def comparar(tarefa, X, y, modelos):
    linhas = []

    for padronizar in (True, False):
        folds = preparar_folds(X, y, padronizar)

        for nome, funcao in modelos.items():
            registros = []

            for X_treino, X_teste, y_treino, y_teste in folds:

                # Sem Z-score o MLP pode saturar ou divergir;
                # os avisos de overflow são esperados.
                with np.errstate(all="ignore"):
                    previsoes = funcao(X_treino, X_teste, y_treino)

                if tarefa == "regressao":
                    if not np.all(np.isfinite(previsoes)):
                        registros = None
                        break
                    m = avaliar_regressao(y_teste, previsoes, X.shape[1])
                    registros.append({
                        "mse": m["mse"],
                        "rmse": m["rmse"],
                        "mae": m["mae"],
                        "r2": m["r2"],
                    })
                else:
                    m = avaliar_classificacao(y_teste, previsoes)
                    registros.append({
                        "acuracia": m["acuracia"],
                        "precisao_macro": m["precisao_macro"],
                        "recall_macro": m["recall_macro"],
                        "f1_macro": m["f1_macro"],
                    })

            linha = {"modelo": nome, "zscore": "com" if padronizar else "sem"}

            if registros is None:
                linha["divergiu"] = "sim"
                print(f"{tarefa} {nome} {linha['zscore']} Z-score: divergiu")
            else:
                resumo = resumir_resultados(registros, nome)
                del resumo["modelo"]
                linha.update(divergiu="nao", **resumo)
                print(
                    f"{tarefa} {nome} {linha['zscore']} Z-score: "
                    + " | ".join(
                        f"{chave[:-6]} {valor:.4f}"
                        for chave, valor in resumo.items()
                        if chave.endswith("_media")
                    ),
                    flush=True
                )

            linhas.append(linha)

    # As configurações que divergiram ficam com as métricas em branco.
    colunas = max(linhas, key=len).keys()
    linhas = [{coluna: linha.get(coluna, "") for coluna in colunas} for linha in linhas]

    return salvar_csv(f"resultados/{tarefa}/zscore.csv", linhas)


def main():
    X, y, _ = carregar_arff("datasets/classificacao/autoUniv-au1-1000.arff")
    print(f"Arquivo gerado: {comparar('classificacao', X, y, MODELOS_CLASSIFICACAO)}")

    X, y, _ = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )
    print(f"Arquivo gerado: {comparar('regressao', X, y, MODELOS_REGRESSAO)}")


if __name__ == "__main__":
    main()
