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

### 7.2 Regressão

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

Os testes kNN podem demorar mais devido à necessidade
de calcular distâncias durante a previsão.

## 10. Resultados experimentais

As tabelas apresentam médias dos cinco folds.

### 10.1 Classificação

| Algoritmo | Acurácia | Precisão macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| kNN Euclidiano | 0.7140 | 0.6089 | 0.5882 | 0.5935 |
| kNN Manhattan | 0.7140 | 0.6089 | 0.5882 | 0.5935 |
| Bayes Univariado | 0.7220 | 0.6080 | 0.5650 | 0.5665 |

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

### 10.2 Regressão

| Algoritmo | MAE | R² | R² ajustado |
|---|---:|---:|---:|
| Regressão Linear | 0.5695 | 0.2882 | 0.2821 |
| kNN Euclidiano | 0.5247 | 0.3582 | 0.3527 |
| kNN Manhattan | 0.5219 | 0.3598 | 0.3543 |

O kNN Manhattan apresentou o menor MAE médio
e o maior R² médio.

A Regressão Linear Múltipla apresentou o menor
tempo médio de previsão entre os regressores.

As diferenças observadas não devem ser interpretadas
automaticamente como estatisticamente significativas.

Os resultados completos, incluindo desvios-padrão
e tempos, estão disponíveis nos arquivos CSV.

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

## 12. Estrutura do projeto

```text
Projeto-IA-AV2-2026.2/
├── datasets/
│   ├── classificacao/
│   └── regressao/
├── src/
│   ├── knn.py
│   ├── classificacao/
│   │   └── bayes.py
│   ├── regressao/
│   │   └── regressao_linear.py
│   └── utils/
│       ├── dados.py
│       ├── exportacao.py
│       ├── metricas.py
│       ├── normalizacao.py
│       └── validacao.py
├── testes/
├── scripts/
│   └── consolidar_resultados.py
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
F1 macro, com diferenças pequenas. Na regressão, os modelos
kNN apresentaram o melhor desempenho.

Os modelos kNN exigiram tempos maiores durante
a previsão, pois comparam as novas instâncias com exemplos
armazenados no conjunto de treinamento.

O Bayes Univariado e a regressão linear apresentaram
tempos de previsão menores, demonstrando a importância
de avaliar simultaneamente qualidade preditiva e
custo computacional.

As conclusões se aplicam aos datasets, configurações
e procedimentos utilizados neste projeto.
