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
- Matplotlib 3.11.2
- Biblioteca CSV nativa do Python
- Git e GitHub

O Matplotlib é utilizado exclusivamente para gerar gráficos.

## 3. Datasets

### 3.1 Classificação: Spambase

Fonte: OpenML — Dataset ID 44.

- Instâncias: 4.601
- Atributos preditores: 57
- Classes: 2
- Tipo de problema: classificação binária

O objetivo é identificar mensagens de spam com base nas
características numéricas extraídas dos e-mails.

Arquivo:

`datasets/classificacao/spambase.arff`

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

O dataset AutoUniv foi utilizado em etapas iniciais do
desenvolvimento, mas não integra os experimentos finais.

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

**Bayes Multivariado**

Classificador probabilístico gaussiano que utiliza
uma matriz de covariância por classe para considerar
relações entre os atributos.

Também utiliza cálculos no domínio logarítmico e
regularização da matriz de covariância.

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

A implementação utiliza operações matriciais do NumPy,
incluindo a resolução numérica do problema de mínimos
quadrados.

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

Foi utilizada validação cruzada com cinco folds.

A semente utilizada na divisão dos dados foi 42.

Na classificação, os folds são estratificados para
preservar aproximadamente a distribuição das classes.

Na regressão, os folds não são estratificados.

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
python -u -m testes.teste_cv_bayes_classificacao univariado
```

Bayes Multivariado:

```bash
python -u -m testes.teste_cv_bayes_classificacao multivariado
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
| kNN Euclidiano | 0.9115 | 0.9094 | 0.9050 | 0.9069 |
| kNN Manhattan | 0.9044 | 0.9041 | 0.8948 | 0.8988 |
| Bayes Univariado | 0.8144 | 0.8267 | 0.8391 | 0.8137 |
| Bayes Multivariado | 0.8276 | 0.8346 | 0.8491 | 0.8266 |

O kNN Euclidiano apresentou a maior acurácia média
e o maior F1 macro médio.

O Bayes Univariado apresentou o menor tempo médio
de previsão entre os classificadores.

### 10.2 Regressão

| Algoritmo | MAE | R² | R² ajustado |
|---|---:|---:|---:|
| Regressão Linear | 0.5698 | 0.2878 | 0.2817 |
| kNN Euclidiano | 0.5229 | 0.3579 | 0.3524 |
| kNN Manhattan | 0.5206 | 0.3618 | 0.3563 |

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

## 12. Gráficos

Para gerar os gráficos:

```bash
python -m scripts.gerar_graficos
```

Os arquivos são salvos em:

`resultados/graficos/`

Gráficos produzidos:

1. Acurácia e F1 macro dos classificadores.
2. Tempo de previsão dos classificadores.
3. MAE dos regressores.
4. R² dos regressores.
5. Tempo de previsão dos regressores.

Os gráficos de tempos utilizam escala logarítmica
para permitir a comparação de valores muito diferentes.

## 13. Estrutura do projeto

```text
Projeto-IA-AV2-2026.2/
├── datasets/
│   ├── classificacao/
│   └── regressao/
├── src/
│   ├── classificacao/
│   │   ├── knn.py
│   │   └── bayes.py
│   ├── regressao/
│   │   ├── knn.py
│   │   └── regressao_linear.py
│   └── utils/
│       ├── dados.py
│       ├── exportacao.py
│       ├── metricas.py
│       ├── normalizacao.py
│       └── validacao.py
├── testes/
├── scripts/
│   ├── consolidar_resultados.py
│   └── gerar_graficos.py
├── resultados/
│   ├── classificacao/
│   ├── regressao/
│   └── graficos/
├── slides/
├── main.py
├── requirements.txt
└── README.md
```

## 14. Conclusão

Nos experimentos realizados, os modelos kNN apresentaram
melhor desempenho preditivo médio tanto na classificação
quanto na regressão.

Entretanto, esses modelos exigiram tempos maiores durante
a previsão, pois comparam as novas instâncias com exemplos
armazenados no conjunto de treinamento.

Os algoritmos bayesianos e a regressão linear apresentaram
tempos de previsão menores, demonstrando a importância
de avaliar simultaneamente qualidade preditiva e
custo computacional.

As conclusões se aplicam aos datasets, configurações
e procedimentos utilizados neste projeto.
