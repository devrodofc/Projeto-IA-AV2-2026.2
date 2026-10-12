import csv
import json
from pathlib import Path

import numpy as np

from src.utils.dados import carregar_arff


MODELO = Path("slides/modelo_interativo.html")
SAIDA = Path("slides/apresentacao_interativa.html")


def ler_resumo(caminho):
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        return {linha["modelo"]: linha for linha in csv.DictReader(arquivo)}


def par(linha, metrica):
    return [
        float(linha[f"{metrica}_media"]),
        float(linha[f"{metrica}_desvio"])
    ]


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
