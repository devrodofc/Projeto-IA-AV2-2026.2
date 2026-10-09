import numpy as np

from src.utils.metricas import (
    mae,
    r2_score,
    r2_ajustado,
    avaliar_regressao
)


def main():

    print("\n========== TESTE MÉTRICAS REGRESSÃO ==========")

    y_real = np.array([
        3.0, 5.0, 7.0, 9.0, 11.0
    ])

    y_previsto = np.array([
        4.0, 5.0, 6.0, 8.0, 10.0
    ])

    quantidade_atributos = 2

    resultados = avaliar_regressao(
        y_real,
        y_previsto,
        quantidade_atributos
    )

    print("\nValores reais:")
    print(y_real)

    print("\nValores previstos:")
    print(y_previsto)

    print("\n========== RESULTADOS ==========")

    print(f"MAE: {resultados['mae']:.4f}")
    print(f"R²: {resultados['r2']:.4f}")
    print(
        f"R² ajustado: "
        f"{resultados['r2_ajustado']:.4f}"
    )

    # Verifica se as funções individuais
    # retornam os mesmos valores.
    assert np.isclose(
        mae(y_real, y_previsto),
        resultados["mae"]
    )

    assert np.isclose(
        r2_score(y_real, y_previsto),
        resultados["r2"]
    )

    assert np.isclose(
        r2_ajustado(
            y_real,
            y_previsto,
            quantidade_atributos
        ),
        resultados["r2_ajustado"]
    )

    # Resultados esperados, calculados manualmente.
    assert np.isclose(resultados["mae"], 0.8)
    assert np.isclose(resultados["r2"], 0.9)
    assert np.isclose(resultados["r2_ajustado"], 0.8)

    print("\nTodos os testes passaram!")


if __name__ == "__main__":
    main()
