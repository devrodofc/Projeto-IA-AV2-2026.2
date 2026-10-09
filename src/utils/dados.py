import csv
import re
import numpy as np
from pathlib import Path


def carregar_arff(caminho, atributo_alvo=None):
    """
    Carrega um dataset ARFF com atributos preditores numéricos.

    Parâmetros:
        caminho: caminho do arquivo ARFF.
        atributo_alvo: nome da coluna que queremos prever.
                       Se não informado, utiliza a última coluna.

    Retorna:
        X: matriz NumPy com os atributos preditores.
        y: vetor NumPy com os valores da variável-alvo.
        nomes_atributos: nomes dos atributos preditores.
    """

    caminho = Path(caminho)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    atributos = []
    tipos = []
    registros = []
    lendo_dados = False

    with caminho.open("r", encoding="utf-8-sig") as arquivo:

        for numero_linha, linha in enumerate(arquivo, start=1):

            linha = linha.strip()

            # Ignora linhas vazias e comentários.
            if not linha or linha.startswith(("%", "%%")):
                continue

            linha_minuscula = linha.lower()

            # Identifica os atributos do ARFF.
            if linha_minuscula.startswith("@attribute"):

                resultado = re.match(
                    r"""@attribute\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+(.+)""",
                    linha,
                    flags=re.IGNORECASE
                )

                if resultado is None:
                    raise ValueError(
                        f"Atributo inválido na linha {numero_linha}"
                    )

                nome = next(
                    valor
                    for valor in resultado.groups()[:3]
                    if valor is not None
                )

                tipo = resultado.group(4).strip()

                atributos.append(nome)
                tipos.append(tipo)

            # Identifica o início dos dados.
            elif linha_minuscula.startswith("@data"):
                lendo_dados = True

            # Lê os registros.
            elif lendo_dados:

                # Esta implementação trabalha com ARFF denso.
                if linha.startswith("{"):
                    raise ValueError(
                        "Formato ARFF esparso não suportado."
                    )

                registro = next(
                    csv.reader([linha], skipinitialspace=True)
                )

                if len(registro) != len(atributos):
                    raise ValueError(
                        f"Linha {numero_linha}: quantidade "
                        "incorreta de valores."
                    )

                registros.append(
                    [valor.strip() for valor in registro]
                )

    if not atributos:
        raise ValueError("Nenhum atributo encontrado.")

    if not registros:
        raise ValueError("Nenhum registro encontrado.")

    # Define qual coluna será prevista.
    if atributo_alvo is None:
        indice_alvo = len(atributos) - 1

    else:
        if atributo_alvo not in atributos:
            raise ValueError(
                f"Atributo-alvo não encontrado: {atributo_alvo}"
            )

        indice_alvo = atributos.index(atributo_alvo)

    # Identifica as colunas preditoras.
    indices_preditores = [
        indice
        for indice in range(len(atributos))
        if indice != indice_alvo
    ]

    # Verifica se os preditores são numéricos.
    tipos_numericos = {
        "numeric",
        "real",
        "integer"
    }

    for indice in indices_preditores:

        if tipos[indice].lower() not in tipos_numericos:
            raise ValueError(
                f"O atributo '{atributos[indice]}' "
                "não é numérico."
            )

    # Verifica valores ausentes.
    for numero_registro, registro in enumerate(registros, start=1):

        if any(valor == "?" for valor in registro):
            raise ValueError(
                f"Valor ausente no registro {numero_registro}."
            )

    # Separa os atributos preditores.
    X = np.array(
        [
            [
                float(registro[indice])
                for indice in indices_preditores
            ]
            for registro in registros
        ],
        dtype=float
    )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Existem valores numéricos inválidos nos preditores."
        )

    # Identifica se a variável-alvo é numérica.
    alvo_numerico = (
        tipos[indice_alvo].lower() in tipos_numericos
    )

    if alvo_numerico:

        # Regressão: mantém valores numéricos.
        y = np.array(
            [
                float(registro[indice_alvo])
                for registro in registros
            ],
            dtype=float
        )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "Existem valores inválidos na variável-alvo."
            )

    else:

        # Classificação: mantém os nomes das classes.
        y = np.array(
            [
                registro[indice_alvo]
                for registro in registros
            ],
            dtype=str
        )

    nomes_atributos = [
        atributos[indice]
        for indice in indices_preditores
    ]

    return X, y, nomes_atributos