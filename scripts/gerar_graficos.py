import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


DIRETORIO_RESULTADOS = Path("resultados")
DIRETORIO_GRAFICOS = DIRETORIO_RESULTADOS / "graficos"

DIRETORIO_GRAFICOS.mkdir(
    parents=True,
    exist_ok=True
)

NOMES_MODELOS = {
    "knn_euclidiana": "kNN Euclidiano",
    "knn_manhattan": "kNN Manhattan",
    "bayes_univariado": "Bayes Univariado",
    "bayes_multivariado": "Bayes Multivariado",
    "linear": "Regressão Linear"
}


def carregar_resumos(categoria):
    """
    Lê os resultados consolidados.
    """

    caminho = (
        DIRETORIO_RESULTADOS
        / f"{categoria}_resumo.csv"
    )

    with caminho.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        registros = list(
            csv.DictReader(arquivo)
        )

    if not registros:
        raise ValueError(
            f"Nenhum resultado encontrado: {caminho}"
        )

    return registros


def nomes_modelos(registros):
    """
    Obtém os nomes utilizados nos gráficos.
    """

    return [
        NOMES_MODELOS[registro["modelo"]]
        for registro in registros
    ]


def valores(registros, coluna):
    """
    Extrai uma coluna numérica do CSV.
    """

    return [
        float(registro[coluna])
        for registro in registros
    ]


def salvar_figura(nome_arquivo):
    """
    Salva o gráfico como PNG.
    """

    caminho = (
        DIRETORIO_GRAFICOS / nome_arquivo
    )

    plt.savefig(
        caminho,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Gráfico criado: {caminho}")


def grafico_classificacao(registros):
    """
    Compara acurácia e F1 macro,
    utilizando seus desvios-padrão.
    """

    nomes = nomes_modelos(registros)
    posicoes = list(range(len(nomes)))

    acuracias = valores(
        registros,
        "acuracia_media"
    )

    desvios_acuracia = valores(
        registros,
        "acuracia_desvio"
    )

    f1 = valores(
        registros,
        "f1_macro_media"
    )

    desvios_f1 = valores(
        registros,
        "f1_macro_desvio"
    )

    plt.figure(figsize=(11, 6))

    pos_acuracia = [
        posicao - 0.10
        for posicao in posicoes
    ]

    pos_f1 = [
        posicao + 0.10
        for posicao in posicoes
    ]

    plt.errorbar(
        pos_acuracia,
        acuracias,
        yerr=desvios_acuracia,
        fmt="o",
        capsize=5,
        markersize=8,
        label="Acurácia"
    )

    plt.errorbar(
        pos_f1,
        f1,
        yerr=desvios_f1,
        fmt="s",
        capsize=5,
        markersize=8,
        label="F1 macro"
    )

    plt.xticks(
        posicoes,
        nomes,
        rotation=15,
        ha="right"
    )

    plt.ylim(0, 1)
    plt.ylabel("Pontuação")
    plt.xlabel("Algoritmo")

    plt.title(
        "Classificação — Acurácia e F1 macro\n"
        "Média e desvio-padrão em 5 folds"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    salvar_figura(
        "classificacao_acuracia_f1.png"
    )


def grafico_regressao(registros, metrica, titulo, arquivo):
    """
    Plota média e desvio-padrão
    de uma métrica de regressão.
    """

    nomes = nomes_modelos(registros)

    medias = valores(
        registros,
        f"{metrica}_media"
    )

    desvios = valores(
        registros,
        f"{metrica}_desvio"
    )

    posicoes = list(
        range(len(nomes))
    )

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        posicoes,
        medias,
        yerr=desvios,
        fmt="o",
        capsize=6,
        markersize=9
    )

    plt.xticks(
        posicoes,
        nomes
    )

    plt.ylabel(metrica.upper())
    plt.xlabel("Algoritmo")

    plt.title(
        f"{titulo}\n"
        "Média e desvio-padrão em 5 folds"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    salvar_figura(arquivo)


def grafico_tempos(registros, titulo, arquivo):
    """
    Compara os tempos médios de previsão.

    Utiliza escala logarítmica porque
    os modelos possuem tempos muito diferentes.
    """

    nomes = nomes_modelos(registros)

    tempos = valores(
        registros,
        "tempo_teste_media"
    )

    if any(tempo <= 0 for tempo in tempos):
        raise ValueError(
            "Os tempos precisam ser positivos."
        )

    posicoes = list(
        range(len(nomes))
    )

    plt.figure(figsize=(11, 6))

    plt.bar(
        posicoes,
        tempos
    )

    plt.yscale("log")

    plt.xticks(
        posicoes,
        nomes,
        rotation=15,
        ha="right"
    )

    plt.ylabel(
        "Tempo médio de previsão (segundos)"
    )

    plt.xlabel("Algoritmo")

    plt.title(
        f"{titulo}\n"
        "Escala logarítmica"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    salvar_figura(arquivo)


def main():

    print(
        "\n========== GERANDO GRÁFICOS =========="
    )

    classificacao = carregar_resumos(
        "classificacao"
    )

    regressao = carregar_resumos(
        "regressao"
    )

    grafico_classificacao(
        classificacao
    )

    grafico_tempos(
        classificacao,
        "Classificação — Tempo de previsão",
        "classificacao_tempo.png"
    )

    grafico_regressao(
        regressao,
        "mae",
        "Regressão — Erro Absoluto Médio",
        "regressao_mae.png"
    )

    grafico_regressao(
        regressao,
        "r2",
        "Regressão — Coeficiente de Determinação",
        "regressao_r2.png"
    )

    grafico_tempos(
        regressao,
        "Regressão — Tempo de previsão",
        "regressao_tempo.png"
    )

    print(
        "\nTodos os gráficos foram gerados!"
    )


if __name__ == "__main__":
    main()
