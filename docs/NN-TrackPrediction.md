# Redes Neurais para Previsão de Trajetória

Notebook referência: [NN-TrackPrediction.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/NN-TrackPrediction.ipynb)


Além de todas as análises feitas, implementamos também uma rede neural que usa informações de um furacão de algumas horas atrás e do presente, e projeta a sua posição futura em coordenadas geográficas.



## Funcionamento da Predição

O input da rede é uma matriz do tipo:

$$\left[\begin{matrix}
x_1^{(t-2)} & x_2^{(t-2)} & ...& x_n^{(t-2)}\\
x_1^{(t-1)} & x_2^{(t-1)} & ...& x_n^{(t-1)}\\
x_1^{(t)} & x_2^{(t)} & ...& x_n^{(t)}\end{matrix}\right]$$

na qual:

- $x_1,~x_2, ~..., ~x_n$ representam os $n$ preditores usados. Testamos dois conjuntos de modelos, o primeiro com 4 preditores (Tempo, Latitude, Longitude e Velocidade de vento) e o segundo com 8 (anteriores mais Temperatura do mar, Umidade, Pressão e Nebulosidade). Fizemos uma comparação da função de perda entre eles para eleger o melhor conjunto de entrada, e usar 8 preditores apresentou melhores resultados.

- $x_i^{(k)}$ representa o preditor $i$ no registro de tempo $k$. Cada registro dos nossos dados está espaçado por 6 horas do próximo e do anterior. Sendo assim, pela matriz acima, usamos um conjunto de 3 registros sequenciais, que representam 18 horas.

Nossa saída é da forma:

O vetor $Y$ geral é composto por:

$[lat^{(t+1)},~ lon^{(t+1)}],$

que representa a latitude e longitude no registro $t+1$ ou seja, + horas depois do último ponto de treinamento.


### Treino, Validação e Teste

Para formatar os dados de treino, teste e validação, tivemos que fazer um tratamento diferenciado e manual, para que o treinamento e previsão ocorresse tempestade por tempestade e não misturasse dados.

Usamos uma divisão de 70% para treino, 20% para validação e 10% para teste. 
Para mais detalhes do processo veja o notebook [NN-TrackPrediction.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/NN-TrackPrediction.ipynb).




## Anatomia das redes e comparação dos modelos

![png](NN-TrackPrediction_files/NN-TrackPrediction_35_2.png)

Fizemos vários modelos com anatomias diferentes apesar de parecidas, elegemos o melhor através do MSE (Erros Médio Quadrático) nos dados de teste e validação.

A anatomia desse modelo campeão é a seguinte:

 - 8 variáveis de input (Tempo, Latitude, Longitude e Velocidade de vento, Temperatura do mar, Umidade, Pressão e Nebulosidade) para cada um dos registros passados. Como usamos 3 registros temos então **24 neurônios** de entrada. 

- 1 única camada interna com 9 neurônios e função de ativação sigmoidal.

- 2 neurônios para a camada de output, um para latitude e outro para longitude, com ativação linear.

- Camada de dropout de 15% para evitar overfitting.


<details>
<summary>Código</summary>
```python

model_892 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(3,8)),
  tf.keras.layers.Dense(9, activation='sigmoid'),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Dense(2, activation='linear')
])
```
</details>


    
![png](NN-TrackPrediction_files/NN-TrackPrediction_26_0.png)


  
## Performance 

Tivemos um R2-score de aproximadamente 0.988 nos dados de teste, um resultado excelente! Abaixo há um código oculto que plota a relação bem próxima de linear entre os dados previstos e reais.


<details>
<summary>Código</summary>

```python
from sklearn.metrics import r2_score
ypred = model_892.predict(xtest2)

lat_r2 = r2_score(ytest2[:,0],ypred[:,0])
lon_r2 = r2_score(ytest2[:,1],ypred[:,1])
tot_r2 = r2_score(ytest2,ypred)

print(f"R2 Latitude Teste - {lat_r2}")
print(f"R2 Longitude Teste - {lon_r2}")
print(f"R2 Total Teste - {tot_r2}")

fig, (ax,ax1) = plt.subplots(1,2,figsize=(18,6))
ax.set_title("Latitude Real vs Prevista (Teste 8-9-2)",fontsize=16)
ax.set_xlabel("Real",fontsize = 13)
ax.set_ylabel("Prevista",fontsize = 13)
ax.scatter(ytest2[:,0],ypred[:,0],alpha = 0.75, color = 'g',label = f"R2 = {round(lat_r2,3)}")
ax.legend(loc='best', fontsize = 13)
ax1.set_title("Longitude Real vs Prevista (Teste 8-9-2)",fontsize=16)
ax1.set_xlabel("Real",fontsize = 13)
ax1.set_ylabel("Prevista",fontsize = 13)
ax1.scatter(ytest2[:,1],ypred[:,1],alpha = 0.75, color = 'g',label = f"R2 = {round(lon_r2,3)}")
ax1.legend(loc='best', fontsize = 13);
plt.savefig('../figs/lat_lon_teste.jpg')
```
</details>
    R2 Latitude Teste - 0.9890722375652974
    R2 Longitude Teste - 0.9879249013965508
    R2 Total Teste - 0.9884985694809241



![png](NN-TrackPrediction_files/NN-TrackPrediction_26_1.png)





## Previsões

Testamos a previsão para algumas tempestades específicas do conjunto de testes e tivemos bons resultados. A função predict_storm recebe um ID de tempestade, e plota as predições além de printar os R2 scores's de Latitude e Longitude nessa ordem.


```python
predict_storm('AL092011')
```

    0.912396866066715
    0.8587266243812284



![png](NN-TrackPrediction_files/NN-TrackPrediction_32_1.png)



```python
predict_storm('AL112015')
```

    0.9650855905600251
    0.9805961966628389



![png](NN-TrackPrediction_files/NN-TrackPrediction_33_1.png)

Para mais detalhes do funcionamento da função predict visite o notebook [NN-TrackPrediction.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/NN-TrackPrediction.ipynb).

