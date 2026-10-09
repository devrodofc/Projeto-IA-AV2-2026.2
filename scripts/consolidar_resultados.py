import csv
from pathlib import Path

from src.utils.exportacao import salvar_csv


DIRETORIO_RESULTADOS = Path("resultados")

MODELOS_CLASSIFICACAO = [
    "knn_euclidiana",
    "knn_manhattan",
    "bayes_univariado",
    "bayes_multivariado"
]

MODELOS_REGRESSAO = [
    "linear",
    "knn_euclidiana",
    "knn_manhattan"
]


def ler_csv(caminho):
    """
    Lê os registros de um arquivo CSV.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    with caminho.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        leitor = csv.DictReader(arquivo)

        if not leitor.fieldnames:
            raise ValueError(
                f"CSV sem cabeçalho: {caminho}"
            )

        registros = list(leitor)

    if not registros:
        raise ValueError(
            f"CSV sem registros: {caminho}"
        )

    return registros


def consolidar(categoria, modelos):
    """
    Reúne os resumos dos modelos de uma categoria.
    
    Também verifica se existem cinco folds
    para cada modelo.
    """

    print(
        f"\n========== {categoria.upper()} =========="
    )

    diretorio = DIRETORIO_RESULTADOS / categoria

    resumos = []

    for modelo in modelos:

        caminho_folds = (
            diretorio / f"{modelo}_folds.csv"
        )

        caminho_resumo = (
            diretorio / f"{modelo}_resumo.csv"
        )

        folds = ler_csv(caminho_folds)

        resumo = ler_csv(caminho_resumo)

        # Verifica a quantidade de folds.
        if len(folds) != 5:
            raise ValueError(
                f"{modelo}: esperados 5 folds, "
                f"encontrados {len(folds)}."
            )

        # Verifica se os folds estão numerados corretamente.
        numeros = sorted(
            int(registro["fold"])
            for registro in folds
        )

        if numeros != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"{modelo}: numeração dos folds inválida."
            )

        # Verifica a identificação dos modelos.
        if any(
            registro["modelo"] != modelo
            for registro in folds
        ):
            raise ValueError(
                f"{modelo}: identificação incorreta nos folds."
            )

        if len(resumo) != 1:
            raise ValueError(
                f"{modelo}: resumo deve possuir uma linha."
            )

        registro_resumo = resumo[0]

        if registro_resumo["modelo"] != modelo:
            raise ValueError(
                f"{modelo}: identificação incorreta no resumo."
            )

        resumos.append(registro_resumo)

        print(
            f"{modelo}: "
            "5 folds encontrados | resumo encontrado"
        )

    # Verifica se todos os resumos possuem
    # as mesmas colunas.
    colunas = set(resumos[0].keys())

    for resumo in resumos:

        if set(resumo.keys()) != colunas:
            raise ValueError(
                "Os resumos possuem colunas diferentes."
            )

    caminho_saida = (
        DIRETORIO_RESULTADOS
        / f"{categoria}_resumo.csv"
    )

    salvar_csv(
        caminho_saida,
        resumos
    )

    print(
        f"\nArquivo consolidado: {caminho_saida}"
    )

    return caminho_saida


def main():

    print(
        "\n========== CONSOLIDAÇÃO DOS EXPERIMENTOS =========="
    )

    consolidar(
        "classificacao",
        MODELOS_CLASSIFICACAO
    )

    consolidar(
        "regressao",
        MODELOS_REGRESSAO
    )

    print(
        "\nTodos os resultados foram consolidados!"
    )


if __name__ == "__main__":
    main()
