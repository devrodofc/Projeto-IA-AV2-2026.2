import time

import numpy as np

from scripts.varrer_hiperparametros import preparar_folds
from src.utils.dados import carregar_arff
from src.utils.metricas import avaliar_classificacao, avaliar_regressao
from src.utils.exportacao import salvar_csv, resumir_resultados
from src.classificacao.perceptron import Perceptron
from src.regressao.mlp import MLP


# Perceptron: todas as combinações de taxa de aprendizado e épocas.
TAXAS_PERCEPTRON = [0.0001, 0.001, 0.01, 0.1, 1.0]
EPOCAS_PERCEPTRON = [1, 5, 10, 25, 50, 100, 200]

# Classe tratada como positiva na sensibilidade, especificidade,
# precisão e F1 do Perceptron: a minoritária (25,9%).
CLASSE_POSITIVA = "class1"

# MLP: parte da configuração usada nos experimentos (a mesma de
# testes/teste_cv_regressao.py) e varia um fator por vez.
BASE_MLP = {
    "camadas": (64,),
    "ativacao": "relu",
    "taxa": 0.01,
    "epocas": 100
}

TOPOLOGIAS_MLP = [
    (8,), (16,), (32,), (64,),
    (16, 8), (32, 16), (64, 32),
    (32, 16, 8)
]
ATIVACOES_MLP = ["sigmoid", "tanh", "relu"]
TAXAS_MLP = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
EPOCAS_MLP = [10, 25, 50, 100, 200, 400]


def cronometrar(funcao, *argumentos):
    inicio = time.perf_counter()
    retorno = funcao(*argumentos)
    return retorno, time.perf_counter() - inicio


def varrer_perceptron():
    X, y, _ = carregar_arff("datasets/classificacao/autoUniv-au1-1000.arff")
    folds = preparar_folds(X, y)
    linhas = []

    for taxa in TAXAS_PERCEPTRON:
        for epocas in EPOCAS_PERCEPTRON:
            registros = []
            matriz = np.zeros((2, 2), dtype=int)

            for X_treino, X_teste, y_treino, y_teste in folds:
                modelo = Perceptron(learning_rate=taxa, n_epochs=epocas)
                _, tempo_treinamento = cronometrar(
                    modelo.fit, X_treino, y_treino
                )
                previsoes, tempo_teste = cronometrar(modelo.predict, X_teste)

                m = avaliar_classificacao(y_teste, previsoes)
                positiva = m["por_classe"][CLASSE_POSITIVA]
                treino = avaliar_classificacao(
                    y_treino, modelo.predict(X_treino)
                )
                matriz += m["matriz_confusao"]

                registros.append({
                    "acuracia": m["acuracia"],
                    "precisao_macro": m["precisao_macro"],
                    "recall_macro": m["recall_macro"],
                    "f1_macro": m["f1_macro"],
                    "sensibilidade": positiva["recall"],
                    "especificidade": positiva["especificidade"],
                    "precisao": positiva["precisao"],
                    "f1": positiva["f1"],
                    "acuracia_treino": treino["acuracia"],
                    "tempo_treinamento": tempo_treinamento,
                    "tempo_teste": tempo_teste,
                })

            resumo = resumir_resultados(registros, "perceptron")
            del resumo["modelo"]

            # Matriz de confusão somada nos 5 folds (cada amostra é
            # testada uma vez). Linhas: real; colunas: previsto.
            linhas.append({
                "learning_rate": taxa,
                "epocas": epocas,
                **resumo,
                "vp": int(matriz[0, 0]),
                "fn": int(matriz[0, 1]),
                "fp": int(matriz[1, 0]),
                "vn": int(matriz[1, 1]),
            })

            print(
                f"Perceptron taxa={taxa} épocas={epocas}: "
                f"acurácia {resumo['acuracia_media']:.4f} | "
                f"F1 macro {resumo['f1_macro_media']:.4f}",
                flush=True
            )

    return salvar_csv(
        "resultados/classificacao/perceptron_hiperparametros.csv", linhas
    )


def configuracoes_mlp():
    """
    Lista as configurações avaliadas, sem repetir a de referência.
    """

    configuracoes = [dict(BASE_MLP, fator="referencia")]

    variacoes = (
        [("topologia", "camadas", v) for v in TOPOLOGIAS_MLP]
        + [("ativacao", "ativacao", v) for v in ATIVACOES_MLP]
        + [("taxa", "taxa", v) for v in TAXAS_MLP]
        + [("epocas", "epocas", v) for v in EPOCAS_MLP]
    )

    for fator, chave, valor in variacoes:
        if valor != BASE_MLP[chave]:
            configuracoes.append(dict(BASE_MLP, fator=fator, **{chave: valor}))

    return configuracoes


def varrer_mlp():
    X, y, _ = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )
    folds = preparar_folds(X, y)
    linhas = []

    for c in configuracoes_mlp():
        registros = []

        for X_treino, X_teste, y_treino, y_teste in folds:
            modelo = MLP(
                hidden_layers=c["camadas"],
                activation=c["ativacao"],
                learning_rate=c["taxa"],
                n_epochs=c["epocas"]
            )

            # Taxas altas podem divergir; os avisos de overflow são esperados.
            with np.errstate(all="ignore"):
                _, tempo_treinamento = cronometrar(
                    modelo.fit, X_treino, y_treino
                )
                previsoes, tempo_teste = cronometrar(modelo.predict, X_teste)
                previsoes_treino = modelo.predict(X_treino)

            if not np.all(np.isfinite(previsoes)):
                registros = None
                break

            m = avaliar_regressao(y_teste, previsoes, X.shape[1])
            treino = avaliar_regressao(y_treino, previsoes_treino, X.shape[1])

            registros.append({
                "mse": m["mse"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "r2_ajustado": m["r2_ajustado"],
                "rmse_treino": treino["rmse"],
                "r2_treino": treino["r2"],
                "tempo_treinamento": tempo_treinamento,
                "tempo_teste": tempo_teste,
            })

        descricao = {
            "fator": c["fator"],
            "camadas": "-".join(str(n) for n in c["camadas"]),
            "ativacao": c["ativacao"],
            "learning_rate": c["taxa"],
            "epocas": c["epocas"],
        }
        rotulo = " ".join(f"{k}={v}" for k, v in descricao.items())

        if registros is None:
            print(f"MLP {rotulo}: divergiu", flush=True)
            linhas.append(dict(descricao, divergiu="sim"))
            continue

        resumo = resumir_resultados(registros, "mlp")
        del resumo["modelo"]
        linhas.append(dict(descricao, divergiu="nao", **resumo))

        print(
            f"MLP {rotulo}: "
            f"RMSE {resumo['rmse_media']:.4f} | R² {resumo['r2_media']:.4f} | "
            f"treino {resumo['tempo_treinamento_media']:.2f}s",
            flush=True
        )

    # As configurações que divergiram ficam com as métricas em branco.
    colunas = max(linhas, key=len).keys()
    linhas = [{coluna: linha.get(coluna, "") for coluna in colunas} for linha in linhas]

    return salvar_csv("resultados/regressao/mlp_topologias.csv", linhas)


def main():
    print(f"Arquivo gerado: {varrer_perceptron()}")
    print(f"Arquivo gerado: {varrer_mlp()}")


if __name__ == "__main__":
    main()
