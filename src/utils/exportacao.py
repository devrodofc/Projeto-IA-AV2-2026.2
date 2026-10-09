import csv
from pathlib import Path

import numpy as np


def salvar_csv(caminho, registros):
    """
    Salva uma lista de dicionários em um arquivo CSV.

    Cada dicionário representa uma linha.
    As chaves representam os nomes das colunas.
    """

    if not registros:
        raise ValueError(
            "É necessário informar pelo menos um registro."
        )

    caminho = Path(caminho)

    # Cria o diretório, caso não exista.
    caminho.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    colunas = list(registros[0].keys())

    for registro in registros:
        if set(registro.keys()) != set(colunas):
            raise ValueError(
                "Todos os registros devem possuir as mesmas colunas."
            )

    with caminho.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=colunas
        )

        escritor.writeheader()
        escritor.writerows(registros)

    return caminho


def resumir_resultados(registros, nome_modelo):
    """
    Calcula média e desvio-padrão das métricas.

    Retorna um dicionário com o nome do modelo
    e as estatísticas calculadas.
    """

    if not registros:
        raise ValueError(
            "Não existem resultados para resumir."
        )

    resumo = {
        "modelo": nome_modelo
    }

    # Identifica as métricas numéricas.
    # Ignora campos descritivos, como modelo e fold.
    campos_ignorados = {
        "modelo",
        "fold"
    }

    metricas = [
        chave
        for chave in registros[0]
        if chave not in campos_ignorados
    ]

    for metrica in metricas:

        valores = np.asarray(
            [
                registro[metrica]
                for registro in registros
            ],
            dtype=float
        )

        if not np.all(np.isfinite(valores)):
            raise ValueError(
                f"A métrica '{metrica}' contém valores inválidos."
            )

        resumo[f"{metrica}_media"] = float(
            np.mean(valores)
        )

        resumo[f"{metrica}_desvio"] = float(
            np.std(valores, ddof=1)
        ) if len(valores) > 1 else 0.0

    return resumo
