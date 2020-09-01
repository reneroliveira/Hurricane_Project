# Hurricane Project

## Pŕe-requisitos

Para garantir a correta execução de todos os notebooks certifique-se que possui todos os pré-requisitos do arquivo requirements.txt instalados. Para instalá-los, execute:

> pip install -r requirements.txt

Além disso, o Notebook Analise_variaveis.ipynb precisa dos pacotes adicionais para realizar alguns plots, mas no próprio notebook há o comando para instalá-los

O download de parte dos dados também precisa ser feito de forma externa através do script download_data.sh. Mas se seguir a sequência de notebooks proposta abaixo, será instruído dentro deles sobre como baixar esses dados.

## Índice

Para a melhor compreensão das etapas, objetivos e conclusões de nosso projeto, siga os arquivos da pasta **Notebooks** na seguinte ordem:

### Dados: Limpeza, Exploração e Visualização
> [Data_Cleaning_and_EDA.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/Data_Cleaning_and_EDA.ipynb)

Nesta etapa descrevemos as fontes de nossos dados, extração e limpeza dos mesmos, além de fazermos a análise exploratória e visualizações importantes.

### Modelos para Predição de Intensidade e Duração de Furacões
> [Analises_variaveis.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/Analises_variaveis.ipynb)

### Análise Temporal do PDI
> [PowerDissipationIndex.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/PowerDissipationIndex.ipynb)

O PDI ou Power Dissipation Index é um índice indicador de destrutibilidade de uma tempestade, o que será melhor explicado no notebook. Nesta estapa fazemos análises temporais desse índice com os dados gerados na etapa de limpeza, de forma a entender a relação de longo prazo entre as quantidades.

### Redes Neurais para Predição de Trajetória
> [NN-TrackPrediction.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/NN-TrackPrediction.ipynb)
Usando os dados gerados na etapa de limpeza, criamos nesta etapa alguns modelos de redes neurais que visavem prever a trajetória futura de um furacão ou tempestade, baseando-se em movimentos e situação climática do passado. O objetivo dos modelos é usar registros de 18h anteriores para prever as coordenadas do centro da tempestade 6 horas depois.
