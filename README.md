# Projeto de Inteligência Artificial — AV2 2026.2

Projeto acadêmico desenvolvido para a disciplina de Inteligência
Artificial Computacional da Universidade de Fortaleza (UNIFOR).

## 1. Objetivo

Implementar e comparar algoritmos de aprendizado de máquina
para problemas de classificação e regressão.

Os algoritmos foram implementados em Python, utilizando NumPy
para operações numéricas, sem utilizar implementações prontas
de aprendizado de máquina, como as disponíveis no scikit-learn.

Os experimentos utilizam validação cruzada com cinco folds,
métricas de avaliação e medição dos tempos de treinamento
e previsão.

Na AV3 o projeto foi estendido com dois modelos neurais,
também implementados manualmente: o Perceptron Simples
(classificação) e o Multi Layer Perceptron — MLP (regressão).

## 2. Tecnologias

- Python 3
- NumPy 2.5.3
- Biblioteca CSV nativa do Python
- Git e GitHub

## 3. Datasets

### 3.1 Classificação: AutoUniv-au1-1000

Fonte: OpenML — Dataset ID 1547.

- Instâncias: 1.000
- Atributos preditores: 20 (binários, valores 0 ou 1)
- Classes: 2 (class1: 25,9%; class2: 74,1%)
- Tipo de problema: classificação binária

Dataset artificial gerado pelo AutoUniv (Ray J. Hickey), que
cria um modelo de classificação aleatório com regras e ruído
controlados e amostra exemplos rotulados a partir dele.

Arquivo:

`datasets/classificacao/autoUniv-au1-1000.arff`

### 3.2 Regressão: Wine Quality

Fonte: OpenML — Dataset ID 287.

- Instâncias: 6.497
- Atributos preditores: 11
- Variável-alvo: quality
- Tipo de problema: regressão

O objetivo é prever a qualidade do vinho utilizando
características físico-químicas.

Arquivo:

`datasets/regressao/wine_quality.arff`

### Observação

O dataset Spambase (OpenML ID 44) foi utilizado em etapas
anteriores do desenvolvimento, mas não integra os
experimentos finais.

## 4. Algoritmos implementados

### 4.1 Classificação

**kNN Euclidiano**

Classifica uma instância considerando a classe mais
frequente entre seus k vizinhos mais próximos.

A distância utilizada é a Euclidiana.

**kNN Manhattan**

Utiliza o mesmo princípio do kNN, mas calcula a
proximidade por meio da distância Manhattan.

Nos experimentos de classificação foi utilizado k = 3.

**Bayes Univariado**

Classificador probabilístico gaussiano que considera
a distribuição de cada atributo por classe, assumindo
independência condicional entre os atributos.

Os cálculos são realizados no domínio logarítmico
para maior estabilidade numérica.

**Perceptron Simples**

Um único neurônio com função de ativação degrau. Os pesos
começam com valores aleatórios e, a cada época, as amostras
são percorridas em ordem embaralhada. Quando a saída está
errada, os pesos são corrigidos pela regra de aprendizado
w ← w + η (d − y) x, em que η é a taxa de aprendizado.

O treinamento termina ao final das épocas ou antes, se uma
época inteira não tiver erros. O modelo só converge quando
as classes são linearmente separáveis.

Hiperparâmetros investigados: taxa de aprendizado e
número de épocas.

### 4.2 Regressão

**kNN Euclidiano**

Prevê o valor de uma instância pela média dos valores
dos k vizinhos mais próximos segundo a distância
Euclidiana.

**kNN Manhattan**

Utiliza a mesma estratégia de previsão, mas com
a distância Manhattan.

Nos experimentos de regressão foi utilizado k = 5.

**Regressão Linear Múltipla**

Estima os coeficientes de uma função linear por meio
do método dos Mínimos Quadrados Ordinários (OLS).

Os coeficientes são obtidos pela equação normal com a
pseudo-inversa de Moore-Penrose:
β = (XᵀX)⁺ Xᵀ y, onde X inclui uma coluna de 1 para
o intercepto.

**Multi Layer Perceptron (MLP)**

Rede neural com uma ou mais camadas ocultas e saída linear.
O treinamento usa gradiente descendente em mini-lotes de
32 amostras: a rede calcula a saída (forward), mede o erro
quadrático médio e propaga o erro da saída para a entrada
(backpropagation) para ajustar pesos e bias.

O alvo é padronizado dentro do `fit` e a previsão é
convertida de volta para a escala original no `predict`.

Hiperparâmetros investigados: quantidade de camadas ocultas,
neurônios por camada, função de ativação (sigmoid, tanh ou
ReLU), taxa de aprendizado e número de épocas.

### 4.3 Origem das implementações

O kNN (classe `KNN` em `src/knn.py`, usada tanto para
classificação quanto para regressão por meio do parâmetro
`task`), o Bayes Univariado (classe `NaiveBayes` em
`src/classificacao/bayes.py`), a Regressão Linear Múltipla
(classe `MRegression`) e o cálculo do R² (função `r_score`)
seguem as implementações apresentadas em aula pela professora.
Ao kNN foi adicionado o parâmetro `distance` para permitir
a distância Manhattan. O código da aula do `NaiveBayes`
contém apenas o treinamento (`fit`); a previsão (`predict`)
foi completada no mesmo estilo, somando o log da priori aos
logs das densidades normais de cada atributo.

O Perceptron Simples (classe `Perceptron` em
`src/classificacao/perceptron.py`) e o MLP (classe `MLP` em
`src/regressao/mlp.py`) foram escritos pela equipe no mesmo
estilo das demais classes, com os métodos `fit` e `predict`.

O `NaiveBayes` não usa regularização na variância. Nos
experimentos nenhum atributo apresenta variância zero dentro
de uma classe (a menor, após a padronização, é 0,63), então
a densidade normal nunca divide por zero.

## 5. Pré-processamento

Os arquivos ARFF são carregados pelo módulo:

`src/utils/dados.py`

A padronização dos atributos é realizada pelo módulo:

`src/utils/normalizacao.py`

Foi utilizada padronização Z-score.

Em cada fold:

1. Os dados são separados em treinamento e teste.
2. A média e o desvio-padrão são calculados apenas no treino.
3. O treino é padronizado.
4. O teste é transformado utilizando os parâmetros do treino.
5. O modelo é treinado.
6. As previsões são realizadas sobre o conjunto de teste.

Esse procedimento evita vazamento de informações
do conjunto de teste durante a padronização.

## 6. Metodologia experimental

Foi utilizada validação cruzada K-fold com cinco folds,
implementada na função `k_fold` (`src/utils/validacao.py`),
no mesmo estilo do `train_test_split` apresentado em aula.

A função embaralha os índices com `np.random.permutation`
(semente 42), divide-os em k partes de tamanho
`ceil(n / k)` e, em cada iteração, utiliza uma parte
como teste e as demais como treinamento.

Os folds não são estratificados, tanto na classificação
quanto na regressão.

Para cada algoritmo são registradas:

- Métricas obtidas em cada fold.
- Tempo de treinamento.
- Tempo de previsão.
- Média das métricas.
- Desvio-padrão amostral das métricas.

Cada algoritmo é avaliado utilizando cinco conjuntos
de treinamento e teste.

## 7. Métricas

### 7.1 Classificação

**Acurácia:** proporção de previsões corretas.

**Precisão macro:** média da precisão calculada
individualmente para cada classe.

**Recall macro:** média do recall calculado
individualmente para cada classe.

**F1 macro:** média do F1 calculado individualmente
para cada classe.

Para o Perceptron também são apresentadas a matriz de
confusão e as métricas da classe positiva (class1, a
minoritária):

**Sensibilidade (recall):** VP / (VP + FN).

**Especificidade:** VN / (VN + FP).

**Precisão:** VP / (VP + FP).

**F1-score:** média harmônica entre precisão e sensibilidade.

### 7.2 Regressão

**MSE:** erro quadrático médio. Penaliza mais os
erros grandes.

**RMSE:** raiz do MSE, na mesma unidade da nota.

**MAE:** erro absoluto médio entre valores reais
e previstos. Valores menores indicam menor erro.

**R²:** coeficiente de determinação. Mede a qualidade
do ajuste em relação à referência baseada na média.

**R² ajustado:** versão do R² que considera a
quantidade de atributos preditores.

O R² ajustado é tradicionalmente associado à
regressão linear. Neste projeto, também foi calculado
para os modelos kNN por padronização da comparação,
sem interpretá-lo como uma medida equivalente de
complexidade entre os diferentes algoritmos.

### 7.3 Tempos

Os tempos de treinamento e previsão são medidos
separadamente.

O tempo de previsão corresponde ao processamento
de todo o conjunto de teste de cada fold.

## 8. Instalação

Clone o repositório:

```bash
git clone git@github.com:devrodofc/Projeto-IA-AV2-2026.2.git
cd Projeto-IA-AV2-2026.2
```

Crie e ative um ambiente virtual no Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## 9. Execução

Execute os comandos a partir da raiz do projeto.

### 9.1 Informações dos datasets

```bash
python main.py
```

Esse comando apresenta as características das duas
bases utilizadas nos experimentos finais.

### 9.2 Classificação

kNN Euclidiano:

```bash
python -u -m testes.teste_cv_knn_classificacao euclidiana
```

kNN Manhattan:

```bash
python -u -m testes.teste_cv_knn_classificacao manhattan
```

Bayes Univariado:

```bash
python -u -m testes.teste_cv_bayes_classificacao
```

Perceptron Simples:

```bash
python -u -m testes.teste_cv_perceptron_classificacao
```

### 9.3 Regressão

Regressão Linear Múltipla:

```bash
python -u -m testes.teste_cv_regressao linear
```

kNN Euclidiano:

```bash
python -u -m testes.teste_cv_regressao knn_euclidiana
```

kNN Manhattan:

```bash
python -u -m testes.teste_cv_regressao knn_manhattan
```

MLP:

```bash
python -u -m testes.teste_cv_regressao mlp
```

Os testes kNN podem demorar mais devido à necessidade
de calcular distâncias durante a previsão.

### 9.4 Varredura de hiperparâmetros das redes neurais

```bash
python -u -m scripts.varrer_redes_neurais
```

O comando avalia, com os mesmos cinco folds, 35 combinações
de taxa de aprendizado e épocas do Perceptron e 20
configurações do MLP (topologia, ativação, taxa de
aprendizado e épocas, variando um fator por vez).

### 9.5 Análise do Z-score

```bash
python -u -m scripts.analisar_zscore
```

O comando avalia todos os algoritmos com e sem a
padronização Z-score.

## 10. Resultados experimentais

As tabelas apresentam médias dos cinco folds.

### 10.1 Classificação

| Algoritmo | Acurácia | Precisão macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| kNN Euclidiano | 0.7140 | 0.6089 | 0.5882 | 0.5935 |
| kNN Manhattan | 0.7140 | 0.6089 | 0.5882 | 0.5935 |
| Bayes Univariado | 0.7220 | 0.6080 | 0.5650 | 0.5665 |
| Perceptron Simples | 0.6700 | 0.5705 | 0.5734 | 0.5704 |

O Bayes Univariado apresentou a maior acurácia média,
enquanto os modelos kNN apresentaram o maior F1 macro médio.
Como as classes são desbalanceadas (74,1% de class2),
o F1 macro é a métrica mais informativa nesta comparação.

Os kNN Euclidiano e Manhattan obtiveram métricas idênticas.
Como os atributos são binários, cada atributo contribui com
uma diferença fixa ou zero, e as duas distâncias tendem a
selecionar os mesmos vizinhos.

O Bayes Univariado apresentou o menor tempo médio
de previsão entre os classificadores.

O Perceptron Simples (η = 1.0, 200 épocas) ficou abaixo
dos demais em acurácia. O dataset não é linearmente
separável, então o modelo não converge e os pesos
continuam oscilando até a última época.

Matriz de confusão do Perceptron, somada nos cinco folds
(classe positiva: class1):

| | Previsto class1 | Previsto class2 |
|---|---:|---:|
| Real class1 | 97 (VP) | 162 (FN) |
| Real class2 | 168 (FP) | 573 (VN) |

| Sensibilidade | Especificidade | Precisão | F1-score |
|---:|---:|---:|---:|
| 0.3738 | 0.7730 | 0.3607 | 0.3649 |

### 10.2 Regressão

| Algoritmo | MSE | RMSE | MAE | R² | R² ajustado |
|---|---:|---:|---:|---:|---:|
| Regressão Linear | 0.5423 | 0.7364 | 0.5695 | 0.2882 | 0.2821 |
| kNN Euclidiano | 0.4890 | 0.6993 | 0.5247 | 0.3582 | 0.3527 |
| kNN Manhattan | 0.4877 | 0.6984 | 0.5219 | 0.3598 | 0.3543 |
| MLP | 0.4794 | 0.6924 | 0.5370 | 0.3709 | 0.3655 |

O kNN Manhattan apresentou o menor MAE médio.

A Regressão Linear Múltipla apresentou o menor
tempo médio de previsão entre os regressores.

O MLP (uma camada oculta de 64 neurônios, ReLU, η = 0.01,
100 épocas) obteve o menor MSE, o menor RMSE e o maior R²,
com MAE um pouco acima dos kNN e previsão em menos de 1 ms.

As diferenças observadas não devem ser interpretadas
automaticamente como estatisticamente significativas.

### 10.3 Hiperparâmetros do Perceptron

F1 macro médio nos cinco folds para cada combinação de
taxa de aprendizado e número de épocas:

| Taxa \ Épocas | 1 | 5 | 10 | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.505 | 0.518 | 0.529 | 0.560 | 0.553 | 0.554 | 0.560 |
| 0.001 | 0.530 | 0.553 | 0.508 | 0.526 | 0.563 | 0.548 | 0.554 |
| 0.01 | 0.564 | 0.514 | 0.523 | 0.521 | 0.557 | 0.553 | 0.566 |
| 0.1 | 0.554 | 0.521 | 0.512 | 0.531 | 0.549 | 0.558 | 0.560 |
| 1 | 0.557 | 0.488 | 0.507 | 0.514 | 0.535 | 0.544 | 0.570 |

Não há tendência clara: o F1 macro varia entre 0,49 e 0,57
sem melhorar de forma consistente com mais épocas ou com
outra taxa, e o desvio-padrão entre folds (até 0,06) é da
mesma ordem das diferenças. Como as classes não são
linearmente separáveis, o Perceptron não converge e o
resultado depende de onde os pesos estavam quando o
treinamento parou. A taxa de aprendizado só muda a escala
das correções em relação aos pesos iniciais.

A configuração usada na comparação (η = 1.0, 200 épocas)
é a de maior F1 macro médio.

### 10.4 Topologias do MLP

Parte-se da configuração usada no trabalho e varia-se um
fator por vez. Médias dos cinco folds:

| Fator variado | Camadas | Ativação | Taxa | Épocas | RMSE | MAE | R² | R² treino | Treino (s) | Teste (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trabalho | 64 | relu | 0.01 | 100 | 0.6924 | 0.5370 | 0.3709 | 0.4489 | 1.04 | 0.0006 |
| Topologia | 8 | relu | 0.01 | 100 | 0.7097 | 0.5525 | 0.3390 | 0.3703 | 0.90 | 0.0002 |
| Topologia | 16 | relu | 0.01 | 100 | 0.7018 | 0.5457 | 0.3537 | 0.3952 | 0.79 | 0.0002 |
| Topologia | 32 | relu | 0.01 | 100 | 0.6979 | 0.5433 | 0.3609 | 0.4195 | 0.89 | 0.0002 |
| Topologia | 16-8 | relu | 0.01 | 100 | 0.6984 | 0.5421 | 0.3600 | 0.4189 | 1.30 | 0.0003 |
| Topologia | 32-16 | relu | 0.01 | 100 | 0.6884 | 0.5323 | 0.3780 | 0.4748 | 1.29 | 0.0005 |
| Topologia | 64-32 | relu | 0.01 | 100 | 0.6926 | 0.5328 | 0.3704 | 0.5316 | 1.49 | 0.0011 |
| Topologia | 32-16-8 | relu | 0.01 | 100 | 0.6932 | 0.5312 | 0.3692 | 0.4802 | 1.85 | 0.0005 |
| Ativação | 64 | sigmoid | 0.01 | 100 | 0.7150 | 0.5547 | 0.3289 | 0.3401 | 1.27 | 0.0011 |
| Ativação | 64 | tanh | 0.01 | 100 | 0.6959 | 0.5374 | 0.3646 | 0.4343 | 1.25 | 0.0012 |
| Taxa | 64 | relu | 0.001 | 100 | 0.7062 | 0.5514 | 0.3456 | 0.3684 | 0.96 | 0.0006 |
| Taxa | 64 | relu | 0.003 | 100 | 0.6955 | 0.5416 | 0.3653 | 0.4024 | 0.90 | 0.0004 |
| Taxa | 64 | relu | 0.03 | 100 | 0.7144 | 0.5526 | 0.3303 | 0.4457 | 0.86 | 0.0004 |
| Taxa | 64 | relu | 0.1 | 100 | 0.7356 | 0.5653 | 0.2896 | 0.4305 | 0.89 | 0.0005 |
| Taxa | 64 | relu | 0.3 | 100 | divergiu | – | – | – | – | – |
| Épocas | 64 | relu | 0.01 | 10 | 0.7130 | 0.5549 | 0.3329 | 0.3570 | 0.07 | 0.0005 |
| Épocas | 64 | relu | 0.01 | 25 | 0.7052 | 0.5505 | 0.3475 | 0.3824 | 0.17 | 0.0005 |
| Épocas | 64 | relu | 0.01 | 50 | 0.6940 | 0.5402 | 0.3681 | 0.4190 | 0.35 | 0.0004 |
| Épocas | 64 | relu | 0.01 | 200 | 0.7036 | 0.5459 | 0.3499 | 0.4707 | 1.43 | 0.0005 |
| Épocas | 64 | relu | 0.01 | 400 | 0.7119 | 0.5515 | 0.3343 | 0.5136 | 2.81 | 0.0004 |

- **Topologia:** mais neurônios reduzem o erro de teste
  (8 → 64 neurônios: RMSE 0,710 → 0,692). A melhor foi 32-16
  (RMSE 0,688). Com 64-32 o R² de treino sobe para 0,532 e o
  de teste não melhora (0,370), sinal de sobreajuste.
- **Ativação:** a ReLU ficou um pouco à frente da tanh; a
  sigmoid aprendeu mais devagar e terminou as 100 épocas pior.
- **Taxa de aprendizado:** valores baixos (0,001) aprendem
  devagar demais para 100 épocas; valores altos (0,03 e 0,1)
  ajustam mais o treino e pioram o teste; com 0,3 o
  treinamento diverge.
- **Épocas:** o erro de teste cai até 100 épocas e volta a
  subir com 200 e 400, enquanto o R² de treino continua
  subindo (0,514 com 400 épocas). O tempo de treinamento
  cresce de forma proporcional ao número de épocas.

A configuração usada na comparação é 64 neurônios, ReLU,
η = 0.01 e 100 épocas. Na varredura, apenas a topologia
32-16 teve erro de teste menor.

As configurações finais foram escolhidas nos mesmos folds
em que são avaliadas, assim como o k do kNN. Os valores
podem estar levemente otimistas.

### 10.5 Impacto da normalização Z-score

Todos os algoritmos foram avaliados com e sem a
padronização, nos mesmos cinco folds e com os mesmos
hiperparâmetros (`python -u -m scripts.analisar_zscore`).

Classificação (AutoUniv-au1-1000):

| Algoritmo | F1 macro com | F1 macro sem | Acurácia com | Acurácia sem |
|---|---:|---:|---:|---:|
| kNN Euclidiano | 0.5935 | 0.5774 | 0.7140 | 0.7050 |
| kNN Manhattan | 0.5935 | 0.5774 | 0.7140 | 0.7050 |
| Bayes Univariado | 0.5665 | 0.5665 | 0.7220 | 0.7220 |
| Perceptron Simples | 0.5704 | 0.4896 | 0.6700 | 0.7120 |

Regressão (Wine Quality):

| Algoritmo | R² com | R² sem | RMSE com | RMSE sem |
|---|---:|---:|---:|---:|
| Regressão Linear | 0.2882 | 0.2882 | 0.7364 | 0.7364 |
| kNN Euclidiano | 0.3582 | 0.1583 | 0.6993 | 0.8007 |
| kNN Manhattan | 0.3598 | 0.1841 | 0.6984 | 0.7884 |
| MLP | 0.3709 | -0.0007 | 0.6924 | 0.8733 |

- **Justificativa:** os atributos do Wine Quality têm escalas
  muito diferentes (o dióxido de enxofre total vai de 6 a 440
  e a densidade de 0,987 a 1,039). No kNN, o atributo de maior
  escala dominaria a distância; no Perceptron e no MLP,
  escalas diferentes desequilibram o ajuste dos pesos. A
  padronização é feita dentro de cada fold, com média e
  desvio-padrão calculados apenas no treino.
- **Bayes Univariado e Regressão Linear:** resultados
  idênticos. A média e a variância por atributo e os
  coeficientes β se ajustam à escala.
- **kNN e MLP na regressão:** sem Z-score o R² do kNN cai de
  0,36 para 0,16–0,18 e o do MLP cai para zero, equivalente a
  prever sempre a média.
- **Classificação:** o efeito é pequeno, pois os atributos já
  são 0 ou 1. O kNN ganha 0,016 de F1 macro. O Perceptron sem
  Z-score tem acurácia maior, mas por prever mais a classe
  majoritária: o F1 macro cai de 0,570 para 0,490.
- **Conclusão:** o uso foi adequado. A padronização poderia
  ser dispensada no Bayes e na Regressão Linear, e o AutoUniv
  poderia ser avaliado sem ela. Para os atributos do vinho
  com valores extremos, uma alternativa é uma escala robusta
  (mediana e intervalo interquartil) ou o Min-Max.

## 11. Arquivos de resultados

Os experimentos geram arquivos CSV individuais:

- `resultados/classificacao/`
- `resultados/regressao/`

Cada algoritmo gera:

- Um arquivo com os resultados dos cinco folds.
- Um arquivo com médias e desvios-padrão.

Para consolidar os resultados:

```bash
python -m scripts.consolidar_resultados
```

O comando produz:

- `resultados/classificacao_resumo.csv`
- `resultados/regressao_resumo.csv`

As redes neurais geram ainda:

- `resultados/classificacao/perceptron_matriz_confusao.csv`:
  matriz de confusão e métricas da classe positiva por fold.
- `resultados/classificacao/perceptron_hiperparametros.csv`:
  varredura de taxa de aprendizado e épocas.
- `resultados/regressao/mlp_topologias.csv`:
  varredura de topologias, ativações, taxas e épocas.
- `resultados/classificacao/zscore.csv` e
  `resultados/regressao/zscore.csv`: métricas com e sem
  a padronização Z-score.

## 12. Estrutura do projeto

```text
Projeto-IA-AV2-2026.2/
├── datasets/
│   ├── classificacao/
│   └── regressao/
├── src/
│   ├── knn.py
│   ├── classificacao/
│   │   ├── bayes.py
│   │   └── perceptron.py
│   ├── regressao/
│   │   ├── mlp.py
│   │   └── regressao_linear.py
│   └── utils/
│       ├── dados.py
│       ├── exportacao.py
│       ├── metricas.py
│       ├── normalizacao.py
│       └── validacao.py
├── testes/
├── scripts/
│   ├── analisar_zscore.py
│   ├── consolidar_resultados.py
│   ├── gerar_apresentacao.py
│   ├── varrer_hiperparametros.py
│   └── varrer_redes_neurais.py
├── resultados/
│   ├── classificacao/
│   └── regressao/
├── slides/
├── main.py
├── requirements.txt
└── README.md
```

## 13. Conclusão

Nos experimentos realizados, na classificação o Bayes
Univariado obteve a maior acurácia e os modelos kNN o maior
F1 macro, com diferenças pequenas. O Perceptron Simples
ficou abaixo dos dois, pois o dataset não é linearmente
separável. Na regressão, os modelos kNN e o MLP
apresentaram o melhor desempenho, com valores próximos.

Os modelos kNN exigiram tempos maiores durante
a previsão, pois comparam as novas instâncias com exemplos
armazenados no conjunto de treinamento.

O MLP concentra o custo no treinamento (cerca de 0,7 s por
fold) e prevê em menos de 1 ms, com qualidade equivalente
à do kNN, que leva cerca de 24 s por fold para prever.

O Bayes Univariado e a regressão linear apresentaram
tempos de previsão menores, demonstrando a importância
de avaliar simultaneamente qualidade preditiva e
custo computacional.

As conclusões se aplicam aos datasets, configurações
e procedimentos utilizados neste projeto.
