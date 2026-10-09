import csv
import numpy as np

from pathlib import Path

from src.utils.exportacao import (
    salvar_csv,
    resumir_resultados
)


def main():

    print("\n========== TESTE DE EXPORTAÇÃO ==========")

    registros = [
        {
            "modelo": "Modelo Exemplo",
            "fold": 1,
            "acuracia": 0.80,
            "f1_macro": 0.75
        },
        {
            "modelo": "Modelo Exemplo",
            "fold": 2,
            "acuracia": 0.90,
            "f1_macro": 0.85
        },
        {
            "modelo": "Modelo Exemplo",
            "fold": 3,
            "acuracia": 0.85,
            "f1_macro": 0.80
        }
    ]

    caminho = Path(
        "resultados/teste_exportacao.csv"
    )

    salvar_csv(
        caminho,
        registros
    )

    assert caminho.exists()

    with caminho.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as arquivo:

        leitor = csv.DictReader(arquivo)
        linhas = list(leitor)

    assert len(linhas) == 3

    assert linhas[0]["modelo"] == "Modelo Exemplo"

    resumo = resumir_resultados(
        registros,
        "Modelo Exemplo"
    )

    print("\nResumo calculado:")

    for chave, valor in resumo.items():
        print(f"{chave}: {valor}")

    assert np.isclose(
        resumo["acuracia_media"],
        0.85
    )

    assert np.isclose(
        resumo["acuracia_desvio"],
        0.05
    )

    assert np.isclose(
        resumo["f1_macro_media"],
        0.80
    )

    assert np.isclose(
        resumo["f1_macro_desvio"],
        0.05
    )

    print("\nArquivo CSV criado:")
    print(caminho)

    print("\nTodos os testes passaram!")


if __name__ == "__main__":
    main()
