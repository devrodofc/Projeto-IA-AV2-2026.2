import numpy as np


def validar_classificacao(y_real, y_previsto):
    """
    Valida os vetores de valores reais e previstos.
    """

    y_real = np.asarray(y_real)
    y_previsto = np.asarray(y_previsto)

    if y_real.ndim != 1 or y_previsto.ndim != 1:
        raise ValueError("Os valores devem ser vetores.")

    if len(y_real) == 0:
        raise ValueError("Os vetores não podem estar vazios.")

    if len(y_real) != len(y_previsto):
        raise ValueError(
            "Os vetores devem possuir o mesmo tamanho."
        )

    return y_real, y_previsto


def matriz_confusao(y_real, y_previsto, classes=None):
    """
    Calcula manualmente a matriz de confusão.

    Linhas: classes reais.
    Colunas: classes previstas.
    """

    y_real, y_previsto = validar_classificacao(
        y_real, y_previsto
    )

    if classes is None:
        classes = np.unique(
            np.concatenate([y_real, y_previsto])
        )
    else:
        classes = np.asarray(classes)

        if len(classes) != len(np.unique(classes)):
            raise ValueError("As classes não podem se repetir.")

        valores = np.unique(
            np.concatenate([y_real, y_previsto])
        )

        if not np.all(np.isin(valores, classes)):
            raise ValueError(
                "Existem valores ausentes na lista de classes."
            )

    matriz = np.zeros(
        (len(classes), len(classes)),
        dtype=int
    )

    for real, previsto in zip(y_real, y_previsto):

        indice_real = np.where(classes == real)[0][0]
        indice_previsto = np.where(classes == previsto)[0][0]

        matriz[indice_real, indice_previsto] += 1

    return matriz, classes


def acuracia(y_real, y_previsto):
    """
    Calcula a proporção de previsões corretas.
    """

    y_real, y_previsto = validar_classificacao(
        y_real, y_previsto
    )

    acertos = np.sum(y_real == y_previsto)

    return float(acertos / len(y_real))


def calcular_vp_fp_fn(y_real, y_previsto, classe_positiva):
    """
    Calcula verdadeiros positivos,
    falsos positivos e falsos negativos.
    """

    y_real, y_previsto = validar_classificacao(
        y_real, y_previsto
    )

    vp = np.sum(
        (y_real == classe_positiva)
        & (y_previsto == classe_positiva)
    )

    fp = np.sum(
        (y_real != classe_positiva)
        & (y_previsto == classe_positiva)
    )

    fn = np.sum(
        (y_real == classe_positiva)
        & (y_previsto != classe_positiva)
    )

    return int(vp), int(fp), int(fn)


def precisao(y_real, y_previsto, classe_positiva):
    """
    Calcula a precisão de uma classe.
    """

    vp, fp, _ = calcular_vp_fp_fn(
        y_real, y_previsto, classe_positiva
    )

    if vp + fp == 0:
        return 0.0

    return vp / (vp + fp)


def recall(y_real, y_previsto, classe_positiva):
    """
    Calcula o recall de uma classe.
    """

    vp, _, fn = calcular_vp_fp_fn(
        y_real, y_previsto, classe_positiva
    )

    if vp + fn == 0:
        return 0.0

    return vp / (vp + fn)


def f1_score(y_real, y_previsto, classe_positiva):
    """
    Calcula o F1-score de uma classe.
    """

    p = precisao(
        y_real, y_previsto, classe_positiva
    )

    r = recall(
        y_real, y_previsto, classe_positiva
    )

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def avaliar_classificacao(y_real, y_previsto):
    """
    Retorna as métricas de classificação
    para todas as classes e as médias macro.
    """

    matriz, classes = matriz_confusao(
        y_real, y_previsto
    )

    resultados_classes = {}

    for classe in classes:

        resultados_classes[str(classe)] = {
            "precisao": precisao(
                y_real, y_previsto, classe
            ),
            "recall": recall(
                y_real, y_previsto, classe
            ),
            "f1": f1_score(
                y_real, y_previsto, classe
            )
        }

    precisao_macro = np.mean([
        resultado["precisao"]
        for resultado in resultados_classes.values()
    ])

    recall_macro = np.mean([
        resultado["recall"]
        for resultado in resultados_classes.values()
    ])

    f1_macro = np.mean([
        resultado["f1"]
        for resultado in resultados_classes.values()
    ])

    return {
        "acuracia": acuracia(y_real, y_previsto),
        "precisao_macro": float(precisao_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "por_classe": resultados_classes,
        "matriz_confusao": matriz,
        "classes": classes
    }


# ============================================================
# MÉTRICAS DE REGRESSÃO
# ============================================================


def validar_regressao(y_real, y_previsto):
    """
    Valida os vetores utilizados nas métricas de regressão.
    """

    import numpy as np

    y_real = np.asarray(y_real, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)

    if y_real.ndim != 1 or y_previsto.ndim != 1:
        raise ValueError(
            "Os valores reais e previstos devem ser vetores."
        )

    if len(y_real) == 0:
        raise ValueError(
            "Os vetores não podem estar vazios."
        )

    if len(y_real) != len(y_previsto):
        raise ValueError(
            "Os vetores devem possuir o mesmo tamanho."
        )

    if not np.all(np.isfinite(y_real)):
        raise ValueError(
            "Os valores reais contêm dados inválidos."
        )

    if not np.all(np.isfinite(y_previsto)):
        raise ValueError(
            "As previsões contêm dados inválidos."
        )

    return y_real, y_previsto


def mae(y_real, y_previsto):
    """
    Calcula o Erro Absoluto Médio (MAE).
    """

    import numpy as np

    y_real, y_previsto = validar_regressao(
        y_real,
        y_previsto
    )

    erros_absolutos = np.abs(
        y_real - y_previsto
    )

    return float(np.mean(erros_absolutos))


def r2_score(y_real, y_previsto):
    """
    Calcula o Coeficiente de Determinação (R²).

    R² = 1 - (SSE / SST)
    """

    import numpy as np

    y_real, y_previsto = validar_regressao(
        y_real,
        y_previsto
    )

    soma_erros_quadrados = np.sum(
        (y_real - y_previsto) ** 2
    )

    media_real = np.mean(y_real)

    soma_total_quadrados = np.sum(
        (y_real - media_real) ** 2
    )

    # Quando todos os valores reais são iguais,
    # a variação total é zero.
    if np.isclose(soma_total_quadrados, 0.0):
        if np.isclose(soma_erros_quadrados, 0.0):
            return 1.0

        return 0.0

    return float(
        1 - soma_erros_quadrados / soma_total_quadrados
    )


def r2_ajustado(y_real, y_previsto, quantidade_atributos):
    """
    Calcula o R² ajustado.

    n = quantidade de instâncias
    p = quantidade de atributos preditores
    """

    if (
        not isinstance(quantidade_atributos, (int, np.integer))
        or isinstance(quantidade_atributos, (bool, np.bool_))
        or quantidade_atributos < 0
    ):
        raise ValueError(
            "A quantidade de atributos deve ser um inteiro não negativo."
        )

    y_real, y_previsto = validar_regressao(
        y_real,
        y_previsto
    )

    n = len(y_real)
    p = quantidade_atributos

    if n <= p + 1:
        raise ValueError(
            "O R² ajustado exige n > p + 1."
        )

    r2 = r2_score(
        y_real,
        y_previsto
    )

    return float(
        1 - (1 - r2) * (n - 1) / (n - p - 1)
    )


def avaliar_regressao(
    y_real,
    y_previsto,
    quantidade_atributos
):
    """
    Calcula todas as métricas de regressão.
    """

    return {
        "mae": mae(y_real, y_previsto),
        "r2": r2_score(y_real, y_previsto),
        "r2_ajustado": r2_ajustado(
            y_real,
            y_previsto,
            quantidade_atributos
        )
    }
