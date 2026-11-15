import csv
import json
from pathlib import Path

import numpy as np

from scripts import varrer_redes_neurais as redes
from src.utils.dados import carregar_arff
from testes import teste_cv_perceptron_classificacao as perceptron


MODELO = Path("slides/modelo_interativo.html")
SAIDA = Path("slides/apresentacao_interativa.html")


def ler_resumo(caminho):
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        return {linha["modelo"]: linha for linha in csv.DictReader(arquivo)}


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def par(linha, metrica):
    return [
        float(linha[f"{metrica}_media"]),
        float(linha[f"{metrica}_desvio"])
    ]


def pares(linha, metricas):
    item = {metrica: par(linha, metrica) for metrica in metricas}
    for chave in ("treino", "teste"):
        coluna = f"tempo_{'treinamento' if chave == 'treino' else 'teste'}"
        if f"{coluna}_media" in linha:
            item[chave] = par(linha, coluna)
    return item


def varredura_perceptron():
    """
    Grade de taxa de aprendizado × épocas do Perceptron.
    """

    metricas = ("acuracia", "precisao_macro", "recall_macro", "f1_macro")
    linhas = ler_csv("resultados/classificacao/perceptron_hiperparametros.csv")

    return {
        "taxas": redes.TAXAS_PERCEPTRON,
        "epocas": redes.EPOCAS_PERCEPTRON,
        "grade": {
            # Chave: índice da taxa | índice das épocas.
            f"{redes.TAXAS_PERCEPTRON.index(float(linha['learning_rate']))}|"
            f"{redes.EPOCAS_PERCEPTRON.index(int(linha['epocas']))}":
                pares(linha, metricas)
            for linha in linhas
        },
        "padrao": {
            "taxa": redes.TAXAS_PERCEPTRON.index(perceptron.LEARNING_RATE),
            "epocas": redes.EPOCAS_PERCEPTRON.index(perceptron.EPOCAS)
        }
    }


def varredura_mlp():
    """
    Para cada fator, os valores avaliados (os outros fatores ficam
    na configuração do trabalho).
    """

    metricas = ("mae", "r2", "r2_ajustado", "r2_treino")
    linhas = ler_csv("resultados/regressao/mlp_topologias.csv")
    base = redes.BASE_MLP

    def chave(camadas, ativacao, taxa, epocas):
        return (
            "-".join(str(n) for n in camadas), ativacao,
            float(taxa), int(epocas)
        )

    por_config = {
        (
            linha["camadas"], linha["ativacao"],
            float(linha["learning_rate"]), int(linha["epocas"])
        ): linha
        for linha in linhas
    }

    fatores = {
        "camadas": ("Topologia", redes.TOPOLOGIAS_MLP),
        "ativacao": ("Ativação", redes.ATIVACOES_MLP),
        "taxa": ("Taxa η", redes.TAXAS_MLP),
        "epocas": ("Épocas", redes.EPOCAS_MLP),
    }
    resultado = {}

    for fator, (nome, valores) in fatores.items():
        itens = []
        divergiram = []

        for valor in valores:
            c = dict(base, **{fator: valor})
            linha = por_config[
                chave(c["camadas"], c["ativacao"], c["taxa"], c["epocas"])
            ]

            # Configurações que divergiram não entram no controle.
            if linha["divergiu"] == "sim":
                divergiram.append(str(valor))
                continue

            item = pares(linha, metricas)
            item["padrao"] = valor == base[fator]
            item["config"] = (
                f"{linha['camadas']} · {c['ativacao']} · "
                f"η = {c['taxa']} · {c['epocas']} épocas"
            )
            item["rotulo"] = (
                linha["camadas"] if fator == "camadas" else str(valor)
            )
            itens.append(item)

        resultado[fator] = {
            "nome": nome,
            "itens": itens,
            "divergiram": divergiram,
            "padrao": [item.pop("padrao") for item in itens].index(True)
        }

    return resultado


def comparacao_zscore():
    """
    Métricas de cada algoritmo com e sem a padronização Z-score.
    """

    resultado = {}

    for tarefa, metricas in (
        ("classificacao", ("acuracia", "f1_macro")),
        ("regressao", ("rmse", "mae", "r2")),
    ):
        resultado[tarefa] = {}

        for linha in ler_csv(f"resultados/{tarefa}/zscore.csv"):
            resultado[tarefa].setdefault(linha["modelo"], {})[linha["zscore"]] = (
                None if linha["divergiu"] == "sim"
                else {m: par(linha, m) for m in metricas}
            )

    return resultado


def main():
    dados = json.loads(
        Path("resultados/hiperparametros.json").read_text(encoding="utf-8")
    )

    cls = ler_resumo("resultados/classificacao_resumo.csv")
    reg = ler_resumo("resultados/regressao_resumo.csv")

    dados["tempos"] = {
        "cls": {
            modelo: {
                "treino": par(linha, "tempo_treinamento"),
                "teste": par(linha, "tempo_teste")
            }
            for modelo, linha in cls.items()
        },
        "reg": {
            modelo: {
                "treino": par(linha, "tempo_treinamento"),
                "teste": par(linha, "tempo_teste")
            }
            for modelo, linha in reg.items()
        },
        "linear": {
            metrica: par(reg["linear"], metrica)
            for metrica in ("mae", "r2", "r2_ajustado")
        }
    }

    dados["perceptron"] = varredura_perceptron()
    dados["mlp"] = varredura_mlp()
    dados["zscore"] = comparacao_zscore()

    X, y, _ = carregar_arff("datasets/classificacao/autoUniv-au1-1000.arff")
    dados["amostras_autouniv"] = [
        {"atributos": [int(v) for v in X[i]], "classe": str(y[i])}
        for i in range(12)
    ]

    _, y_vinho, _ = carregar_arff(
        "datasets/regressao/wine_quality.arff",
        atributo_alvo="quality"
    )
    notas, contagens = np.unique(y_vinho.astype(int), return_counts=True)
    dados["qualidade"] = {
        "notas": notas.tolist(),
        "contagens": contagens.tolist()
    }

    conteudo = MODELO.read_text(encoding="utf-8").replace(
        "__DADOS__",
        json.dumps(dados, separators=(",", ":"))
    )

    # Documento completo: sem o doctype o navegador entra em modo de
    # compatibilidade (tabelas não herdam cor e a impressão quebra).
    html = (
        "<!doctype html>\n<html lang=\"pt-BR\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, "
        "initial-scale=1, viewport-fit=cover\">\n"
        "<style>body{margin:0}img{max-width:100%}"
        "[hidden]{display:none!important}</style>\n"
        "</head>\n<body>\n"
        f"{conteudo}\n</body>\n</html>\n"
    )
    SAIDA.write_text(html, encoding="utf-8")
    print(f"Apresentação gerada: {SAIDA}")


if __name__ == "__main__":
    main()
