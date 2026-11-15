import numpy as np

from src.utils.metricas import avaliar_classificacao, avaliar_regressao
from src.classificacao.perceptron import Perceptron
from src.regressao.mlp import MLP


def testar_perceptron():

    print("\n========== TESTE PERCEPTRON SIMPLES ==========")

    gerador = np.random.default_rng(0)

    # Classes linearmente separáveis: o Perceptron deve convergir.
    X = gerador.normal(size=(400, 3))

    y = np.where(
        X @ np.array([1.0, -2.0, 0.5]) + 0.3 > 0,
        "class2",
        "class1"
    )

    modelo = Perceptron(learning_rate=0.1, n_epochs=100)

    modelo.fit(X, y)

    metricas = avaliar_classificacao(y, modelo.predict(X))

    print(f"Acurácia: {metricas['acuracia']:.4f}")
    print("Matriz de confusão:")
    print(metricas["matriz_confusao"])

    assert metricas["acuracia"] == 1.0

    for classe in ("class1", "class2"):
        assert metricas["por_classe"][classe]["especificidade"] == 1.0

    # Especificidade calculada manualmente:
    # reais negativos = 3, previstos como negativos = 2.
    y_real = np.array(["p", "p", "n", "n", "n"])
    y_previsto = np.array(["p", "n", "n", "n", "p"])

    metricas = avaliar_classificacao(y_real, y_previsto)

    assert np.isclose(
        metricas["por_classe"]["p"]["especificidade"],
        2 / 3
    )

    assert np.isclose(metricas["por_classe"]["p"]["recall"], 0.5)


def testar_mlp():

    print("\n========== TESTE MLP ==========")

    gerador = np.random.default_rng(0)

    # Relação não linear entre os atributos e o alvo.
    X = gerador.uniform(-2, 2, size=(2000, 2))

    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 5

    for ativacao in ("sigmoid", "tanh", "relu"):

        modelo = MLP(
            hidden_layers=(16, 8),
            activation=ativacao,
            learning_rate=0.05,
            n_epochs=200
        )

        modelo.fit(X, y)

        metricas = avaliar_regressao(y, modelo.predict(X), X.shape[1])

        print(
            f"{ativacao}: RMSE {metricas['rmse']:.4f} | "
            f"R² {metricas['r2']:.4f}"
        )

        assert metricas["r2"] > 0.98

        assert np.isclose(
            metricas["rmse"],
            np.sqrt(metricas["mse"])
        )

    # Confere o backpropagation com o gradiente numérico.
    modelo = MLP(hidden_layers=(5, 4), learning_rate=0.0, n_epochs=1)

    modelo.fit(X, y)

    d = ((y - modelo._y_mean) / modelo._y_std).reshape(-1, 1)

    def erro():
        return np.mean((modelo.forward(X)[-1] - d) ** 2)

    A = modelo.forward(X)
    delta = 2 * (A[-1] - d) / len(X)

    for camada in reversed(range(len(modelo.W))):

        gradiente = (A[camada].T @ delta)[0, 0]

        if camada > 0:
            delta = (
                (delta @ modelo.W[camada].T)
                * modelo.derivative(A[camada])
            )

        pesos = modelo.W[camada]
        original = pesos[0, 0]

        pesos[0, 0] = original + 1e-6
        erro_mais = erro()

        pesos[0, 0] = original - 1e-6
        erro_menos = erro()

        pesos[0, 0] = original

        assert np.isclose(
            gradiente,
            (erro_mais - erro_menos) / 2e-6,
            atol=1e-7
        )

    print("Gradientes do backpropagation conferidos.")


def main():

    testar_perceptron()
    testar_mlp()

    print("\nTodos os testes passaram!")


if __name__ == "__main__":
    main()
