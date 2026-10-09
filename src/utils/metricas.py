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
