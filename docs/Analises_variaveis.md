# Aplicação de Modelos de Regressão

[Notebook referência](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/Analises_variaveis.ipynb)

Nesta página iremos mostrar os diversos algoritmos aplicados aos dados visando ajustar uma boa regressão que ajudasse a prever as velocidades dos ventos de um determinado evento (Tropical Storm ou Hurricane) ou a duração dos mesmos.
Algumas pequenas transformações foram necessárias para ajuste das variáveis preditoras a serem consideradas em cada modelo. Para detalhes sobre o carregamento e preparação dos dados, veja o notebook referência.


## Modelos


### Análises Iniciais

Segue abaixo o resultado da aplicação de uma regressão linear simples usando como variável alvo a velocidade de vento, dado usado em vários indicadores de destrutibilidade. Interessante notar os parâmetros com coeficientes positivos.


<details>
<summary>Código</summary>
```python
X_train2 = sm.add_constant(X_train) #np.array(X_train).reshape(X_train.shape[0],1)
OLS_obj = OLS(y_train_mw, X_train2)
OLSModel = OLS_obj.fit()

r2_train = OLSModel.rsquared

print(f'R^2_train = {r2_train}')

print(f'Parâmetro_const  = {OLSModel.params[0]}')
print(f'Parâmetro_Year  = {OLSModel.params[1]}')
print(f'Parâmetro_Month  = {OLSModel.params[2]}')
print(f'Parâmetro_Latitude  = {OLSModel.params[3]}')
print(f'Parâmetro_Longitude  = {OLSModel.params[4]}')
print(f'Parâmetro_sst  = {OLSModel.params[5]}')
print(f'Parâmetro_rhum  = {OLSModel.params[6]}')
print(f'Parâmetro_wspd  = {OLSModel.params[7]}')
print(f'Parâmetro_slp  = {OLSModel.params[8]}')
print(f'Parâmetro_cldc  = {OLSModel.params[9]}')

```
</details>

    R^2_train = 0.02385621171100183
    Parâmetro_const  = 47.35464784180999
    Parâmetro_Year  = 0.09050506606867095
    Parâmetro_Month  = -0.05872870746380546
    Parâmetro_Latitude  = -0.08576970286238517
    Parâmetro_Longitude  = 1.880119707824508
    Parâmetro_sst  = 0.15179194438994867
    Parâmetro_rhum  = 0.028283243122749335
    Parâmetro_wspd  = 0.028283243122749446
    Parâmetro_slp  = 0.14971356534654948
    Parâmetro_cldc  = -1.5161434590996923


Os códigos abaixos nos gera uma visualização que pode trazer insights a respeito da relação entre as variáveis. A escolha das variáveis preditoras que servem de entrada para os modelos mais a frente foram pensadas também pela observação destes gráficos.


<details>
<summary>Código</summary>
```python
df = pd.concat([X_train, y_train_mw, y_train_mp], axis=1)
scatter_matrix(df, alpha=0.8, figsize=(15, 15), diagonal='kde');

```
</details>

![png](Analises_variaveis_files/Analises_variaveis_13_0.png)


<details>
<summary>Código</summary>
```python
# A princípio, não queremos que se faça alguma previsão com base no valor numérico do ano
# Além disso, a variável wspd está altamente correlacionada com a rhum, podendo ser mantida apenas a última
X_train = data_atl_merged.drop(['Year', 'wspd'], 1)

#   Mês      Latitude    Longitude    Temperatura, Umidade, Sea Level Pressure, Cloudiness]
# ['Month', 'Latitude', 'Longitude', 'sst',       'rhum',  'slp',              'cldc']

fig, ax = plt.subplots(1,7)#, figsize=(16,10))
fig.suptitle('Velocidade Máxima vs Variáveis Preditoras (1950-2015)', fontsize=28, y=1.06)

ax[0].scatter(X_train['Month'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 
ax[1].scatter(X_train['Latitude'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 
ax[2].scatter(X_train['Longitude'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 
ax[3].scatter(X_train['sst'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 
ax[4].scatter(X_train['rhum'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 
ax[5].scatter(X_train['slp'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 
ax[6].scatter(X_train['cldc'], X_train['Maximum Wind'], alpha = 0.5, ls = '--') 

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```
</details>

![png](Analises_variaveis_files/Analises_variaveis_15_0.png)




Uma primeira tentativa de ajuste foi feito através da centralização das variáveis preditoras em relação à média, adicionando também termos polinomiais de segunda ordem. No entanto, os resultados do ajuste não mostraram ganhos significativos para o modelo de Regressão Linear Múltipla, e até prejudicaram modelos mais complexos, como Random Forest, Multi Layer Perceptron, entre outros utilizados mais a frente. Detalhes desta parte do código acesse o notebook referência, no link do início desta página.







### Random Forest

Primeira tentativa de ajuste já nos parece promissor em relação aos demais.


<details>
<summary>Código</summary>
```python
#X_train, y_train_mw = make_regression(n_features=7, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(n_estimators=7, max_depth=20, random_state=0)
regr.fit(X_train, y_train_mw)
print(regr.score(X_train, y_train_mw))
print(regr.score(X_test, y_test_mw))

```
</details>

    0.8272253680614361
    0.4358801020027704




### Multi Layer Perceptron

<details>
<summary>Código</summary>
```python
#X_train, y_train_mw = make_regression(n_samples=200, random_state=1)
regr = MLPRegressor(hidden_layer_sizes=(100,2), random_state=1, max_iter=1000, solver='lbfgs').fit(X_train, y_train_mw)
#regr.predict(X_test[:2])
print(regr.score(X_train, y_train_mw))
print(regr.score(X_test, y_test_mw))

```
</details>

    0.09520915346048042
    0.08725614810614979



## Modelos com Separação em Conjuntos de Treino e Teste

Separamos os dados em conjuntos de treino e de teste. Deste modo, podemos ajustar o algoritmo utilizando os dados de treino, e tentar utilizar esses dados de teste para previsão de outros dados, inclusive futuros.


```python
X_train = data_atl_merged.drop(['ID', 'Name', 'Date', 'Time', 'Event', 'Status', 'Maximum Wind', 'Minimum Pressure', 'Date_c', 'Day', 'Latitude_c', 'Longitude_c', 'Duration', 'Year', 'wspd'], 1)
y_train_mw = data_atl_merged['Maximum Wind']
#print(len(X_train))
#X_train.head()

X_train, X_test, y_train_mw, y_test_mw = train_test_split(X_train, y_train_mw, random_state=1)
```

### Regressão Linear


```python
X_train2 = sm.add_constant(X_train) #np.array(X_train).reshape(X_train.shape[0],1)
X_test2 = sm.add_constant(X_test) #np.array(X_train).reshape(X_train.shape[0],1)
OLS_obj = OLS(y_train_mw, X_train2)
OLSModel = OLS_obj.fit()

r2_train = OLSModel.rsquared
r2_test = 1 - ((OLSModel.predict(X_test2)-y_test_mw)*(OLSModel.predict(X_test2)-y_test_mw)).sum() / ((y_test_mw.mean()-y_test_mw)*(y_test_mw.mean()-y_test_mw)).sum()
print(f'R^2_train = {r2_train}')
print(f'R^2_test  = {r2_test}')
'''
print(f'Parâmetro_const  = {OLSModel.params[0]}')
print(f'Parâmetro_Month  = {OLSModel.params[1]}')
print(f'Parâmetro_Latitude  = {OLSModel.params[2]}')
print(f'Parâmetro_Longitude  = {OLSModel.params[3]}')
print(f'Parâmetro_sst  = {OLSModel.params[4]}')
print(f'Parâmetro_rhum  = {OLSModel.params[5]}')
print(f'Parâmetro_slp  = {OLSModel.params[6]}')
print(f'Parâmetro_cldc  = {OLSModel.params[7]}')

print(f'Parâmetro_Month^2  = {OLSModel.params[8]}')
print(f'Parâmetro_Latitude^2  = {OLSModel.params[9]}')
print(f'Parâmetro_Longitude^2  = {OLSModel.params[10]}')
print(f'Parâmetro_sst^2  = {OLSModel.params[11]}')
print(f'Parâmetro_rhum^2  = {OLSModel.params[12]}')
print(f'Parâmetro_slp^2  = {OLSModel.params[13]}')
print(f'Parâmetro_cldc^2  = {OLSModel.params[14]}')
'''
```

    R^2_train = 0.019806236602926464
    R^2_test  = 0.01874952522766249





    "\nprint(f'Parâmetro_const  = {OLSModel.params[0]}')\nprint(f'Parâmetro_Month  = {OLSModel.params[1]}')\nprint(f'Parâmetro_Latitude  = {OLSModel.params[2]}')\nprint(f'Parâmetro_Longitude  = {OLSModel.params[3]}')\nprint(f'Parâmetro_sst  = {OLSModel.params[4]}')\nprint(f'Parâmetro_rhum  = {OLSModel.params[5]}')\nprint(f'Parâmetro_slp  = {OLSModel.params[6]}')\nprint(f'Parâmetro_cldc  = {OLSModel.params[7]}')\n\nprint(f'Parâmetro_Month^2  = {OLSModel.params[8]}')\nprint(f'Parâmetro_Latitude^2  = {OLSModel.params[9]}')\nprint(f'Parâmetro_Longitude^2  = {OLSModel.params[10]}')\nprint(f'Parâmetro_sst^2  = {OLSModel.params[11]}')\nprint(f'Parâmetro_rhum^2  = {OLSModel.params[12]}')\nprint(f'Parâmetro_slp^2  = {OLSModel.params[13]}')\nprint(f'Parâmetro_cldc^2  = {OLSModel.params[14]}')\n"




```python
fig, ax = plt.subplots(1,2)#, figsize=(16,10))
#fig.suptitle('Velocidade Máxima vs Pressão Mínima (1851-2015)', fontsize=28, y=1.06)

ax[0].scatter(X_train['sst'], y_train_mw, alpha=0.5, label=r'$Dados$ $de$ $Treino$')
ax[0].scatter(X_train['sst'], OLSModel.predict(X_train2), alpha=0.5, label=r'$Previsão$')
ax[1].scatter(X_test['sst'], y_test_mw, alpha=0.5, label=r'$Dados$ $de$ $Teste$')
ax[1].scatter(X_test['sst'], OLSModel.predict(X_test2), alpha=0.5, label=r'$Previsão$')

ax[0].tick_params(labelsize=24)
ax[0].set_title(f'Previsão de Maximal Wind por Regressão Linear (Treino)', fontsize=24)
ax[0].set_xlabel(r'$sst$', fontsize=16)
ax[0].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[0].legend(loc='best', fontsize=12);

ax[1].tick_params(labelsize=24)
ax[1].set_title(f'Previsão de Maximal Wind por Regressão Linear (Teste)', fontsize=24)
ax[1].set_xlabel(r'$sst$', fontsize=16)
ax[1].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[1].legend(loc='best', fontsize=12);

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```


![png](Analises_variaveis_files/Analises_variaveis_34_0.png)


### Random Forest

Pelos ajustes anteriores, vimos que esse algoritmo promove um bom ajuste nos dados. Um novo ajuste com aplicação de parâmetros melhor sintonizados com os dados é buscado pelo código abaixo.


```python
# Parâmetros com bom ajuste para Random Forest: n_estimators = 50, max_depth = 75
for i in [25, 50, 75, 100, 125]:
    for j in [25, 50, 75, 100, 125]:
        regr_rf = RandomForestRegressor(n_estimators=i, max_depth=j, random_state=0, oob_score=True, bootstrap = True)
        regr_rf.fit(X_train, y_train_mw)
        print(f'\n n_estimators={i}, max_depth={j}')
        print(regr_rf.score(X_train, y_train_mw))
        print(regr_rf.score(X_test, y_test_mw))

```

    
     n_estimators=25, max_depth=25
    0.9109127406378082
    0.5049385543175804
    
     n_estimators=25, max_depth=50
    0.9231191839999706
    0.5083859136052509
    
     n_estimators=25, max_depth=75
    0.9231191839999706
    0.5083859136052509
    
     n_estimators=25, max_depth=100
    0.9231191839999706
    0.5083859136052509
    
     n_estimators=25, max_depth=125
    0.9231191839999706
    0.5083859136052509
    
     n_estimators=50, max_depth=25
    0.9177312843005485
    0.5190516352697632
    
     n_estimators=50, max_depth=50
    0.9298001112559356
    0.5227899508473133
    
     n_estimators=50, max_depth=75
    0.9298001112559356
    0.5227899508473133
    
     n_estimators=50, max_depth=100
    0.9298001112559356
    0.5227899508473133
    
     n_estimators=50, max_depth=125
    0.9298001112559356
    0.5227899508473133
    
     n_estimators=75, max_depth=25
    0.9198026089627763
    0.5227499387064152
    
     n_estimators=75, max_depth=50
    0.9323017737677339
    0.5256624725114608
    
     n_estimators=75, max_depth=75
    0.9323017737677339
    0.5256624725114608
    
     n_estimators=75, max_depth=100
    0.9323017737677339
    0.5256624725114608
    
     n_estimators=75, max_depth=125
    0.9323017737677339
    0.5256624725114608
    
     n_estimators=100, max_depth=25
    0.9198383784088979
    0.5224954900261807
    
     n_estimators=100, max_depth=50
    0.9336736647334395
    0.5269361337556209
    
     n_estimators=100, max_depth=75
    0.9336736647334395
    0.5269361337556209
    
     n_estimators=100, max_depth=100
    0.9336736647334395
    0.5269361337556209
    
     n_estimators=100, max_depth=125
    0.9336736647334395
    0.5269361337556209
    
     n_estimators=125, max_depth=25
    0.9204444093451314
    0.5234545174002654
    
     n_estimators=125, max_depth=50
    0.9344355008748928
    0.5280216690661794
    
     n_estimators=125, max_depth=75
    0.9344355008748928
    0.5280216690661794
    
     n_estimators=125, max_depth=100
    0.9344355008748928
    0.5280216690661794
    
     n_estimators=125, max_depth=125
    0.9344355008748928
    0.5280216690661794


O R2 Score obtido abaixo mostra o melhor ajuste do modelo quando tentamos prever a Velocidade Máxima Sustentada pelo algoritmo do Random Forest. O ajuste aos dados de treino ficam


```python
regr_rf = RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf.fit(X_train, y_train_mw)
print(regr_rf.score(X_train, y_train_mw))
print(regr_rf.score(X_test, y_test_mw))
```

    0.9298001112559356
    0.5227899508473133



```python
X_train_red = X_train.drop(['sst', 'rhum', 'slp', 'cldc'], 1)
X_test_red = X_test.drop(['sst', 'rhum', 'slp', 'cldc'], 1)

```

Retirando os dados climáticos, observamos que o ajuste fica bem pior, mostrando a importância dos mesmos para a predição


```python
regr_rf_red = RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf_red.fit(X_train_red, y_train_mw)
print(regr_rf_red.score(X_train_red, y_train_mw))
print(regr_rf_red.score(X_test_red, y_test_mw))
```

    0.8598571752407663
    0.06143697855472363



```python
X_train_red = X_train.drop(['Month', 'Latitude', 'Longitude'], 1)
X_test_red = X_test.drop(['Month', 'Latitude', 'Longitude'], 1)

```


```python
regr_rf_red = RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf_red.fit(X_train_red, y_train_mw)
print(regr_rf_red.score(X_train_red, y_train_mw))
print(regr_rf_red.score(X_test_red, y_test_mw))
```

    0.9072010288584739
    0.3833348196160016



```python
#['Month', 'Latitude', 'Longitude', 'sst', 'rhum', 'slp', 'cldc']
X_train_red = X_train.drop(['Month'], 1)
X_test_red = X_test.drop(['Month'], 1)

```


```python
regr_rf_red = RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf_red.fit(X_train_red, y_train_mw)
print(regr_rf_red.score(X_train_red, y_train_mw))
print(regr_rf_red.score(X_test_red, y_test_mw))
```

    0.9230537875398801
    0.4840455607978509


Ajuste da predição em relação à variável sst (temperatura mensal média)


```python
fig, ax = plt.subplots(1,2)#, figsize=(16,10))
#fig.suptitle('Velocidade Máxima vs Pressão Mínima (1851-2015)', fontsize=28, y=1.06)

ax[0].scatter(X_train['sst'], y_train_mw, alpha=0.5, label=r'$Dados$ $de$ $Treino$')
ax[0].scatter(X_train['sst'], regr_rf.predict(X_train), alpha=0.5, label=r'$Previsão$')
ax[1].scatter(X_test['sst'], y_test_mw, alpha=0.5, label=r'$Dados$ $de$ $Teste$')
ax[1].scatter(X_test['sst'], regr_rf.predict(X_test), alpha=0.5, label=r'$Previsão$')

ax[0].tick_params(labelsize=24)
ax[0].set_title(f'Previsão de Maximal Wind por Random Forest (Treino)', fontsize=24)
ax[0].set_xlabel(r'$sst$', fontsize=16)
ax[0].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[0].legend(loc='best', fontsize=12);

ax[1].tick_params(labelsize=24)
ax[1].set_title(f'Previsão de Maximal Wind por Random Forest (Teste)', fontsize=24)
ax[1].set_xlabel(r'$sst$', fontsize=16)
ax[1].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[1].legend(loc='best', fontsize=12);

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```


![png](Analises_variaveis_files/Analises_variaveis_48_0.png)


### Demais Previsões com Random Forest (Melhor Ajuste)

Adicionando as variáveis Ano e Dia, conseguimos melhorar significativamente a capacidade de previsão do nosso modelo.
Se adicionarmos primeiramente apenas a variável Ano, percebemos que cada variável contribui um pouco para a melhoria da previsão.


```python
data_train_sd = data_atl_merged.drop(['ID', 'Name', 'Date', 'Time', 'Event', 'Status', 'Maximum Wind', 'Minimum Pressure', 'Date_c', 'Latitude_c', 'Longitude_c', 'Duration', 'wspd', 'Day'], 1)
data_train_mw_sd = data_atl_merged['Maximum Wind']
#print(len(data_train))
#data_train.head()
data_train_sd, data_test_sd, data_train_mw_sd, data_test_mw_sd = train_test_split(data_train_sd, data_train_mw_sd, random_state=1)

```


```python
regr_rf2_sd = RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf2_sd.fit(data_train_sd, data_train_mw_sd)
print(regr_rf2_sd.score(data_train_sd, data_train_mw_sd))
print(regr_rf2_sd.score(data_test_sd, data_test_mw_sd))
```

    0.9502630426894276
    0.6609429559863542



```python
data_train = data_atl_merged.drop(['ID', 'Name', 'Date', 'Time', 'Event', 'Status', 'Maximum Wind', 'Minimum Pressure', 'Date_c', 'Latitude_c', 'Longitude_c', 'Duration', 'wspd'], 1)
data_train_mw = data_atl_merged['Maximum Wind']
#print(len(data_train))
#data_train.head()
data_train, data_test, data_train_mw, data_test_mw = train_test_split(data_train, data_train_mw, random_state=1)
```

Ajuste fino dos parâmetros do Random Forest


```python
for i in [25, 50, 75, 100, 125]:
    for j in [25, 50, 75, 100, 125]:
        regr_rf2 = RandomForestRegressor(n_estimators=i, max_depth=j, random_state=0, oob_score=True, bootstrap = True)
        regr_rf2.fit(data_train, data_train_mw)
        print(f'\n n_estimators={i}, max_depth={j}')
        print(regr_rf2.score(data_train, data_train_mw))
        print(regr_rf2.score(data_test, data_test_mw))
```

    
     n_estimators=25, max_depth=25
    0.957264562824329
    0.7444676687346707
    
     n_estimators=25, max_depth=50
    0.9582917968925033
    0.7442935073801389
    
     n_estimators=25, max_depth=75
    0.9582917968925033
    0.7442935073801389
    
     n_estimators=25, max_depth=100
    0.9582917968925033
    0.7442935073801389
    
     n_estimators=25, max_depth=125
    0.9582917968925033
    0.7442935073801389
    
     n_estimators=50, max_depth=25
    0.9617003376388379
    0.7583441990969142
    
     n_estimators=50, max_depth=50
    0.9629951225598822
    0.7593567448937373
    
     n_estimators=50, max_depth=75
    0.9629951225598822
    0.7593567448937373
    
     n_estimators=50, max_depth=100
    0.9629951225598822
    0.7593567448937373
    
     n_estimators=50, max_depth=125
    0.9629951225598822
    0.7593567448937373
    
     n_estimators=75, max_depth=25
    0.9637876668338223
    0.7615184996888411
    
     n_estimators=75, max_depth=50
    0.9651229272756359
    0.7623946732426374
    
     n_estimators=75, max_depth=75
    0.9651229272756359
    0.7623946732426374
    
     n_estimators=75, max_depth=100
    0.9651229272756359
    0.7623946732426374
    
     n_estimators=75, max_depth=125
    0.9651229272756359
    0.7623946732426374
    
     n_estimators=100, max_depth=25
    0.9643667689055675
    0.7618762745120209
    
     n_estimators=100, max_depth=50
    0.9657927394363737
    0.7630103360785616
    
     n_estimators=100, max_depth=75
    0.9657927394363737
    0.7630103360785616
    
     n_estimators=100, max_depth=100
    0.9657927394363737
    0.7630103360785616
    
     n_estimators=100, max_depth=125
    0.9657927394363737
    0.7630103360785616
    
     n_estimators=125, max_depth=25
    0.9647081307547872
    0.762938280257875
    
     n_estimators=125, max_depth=50
    0.9661419630929642
    0.7643016255435418
    
     n_estimators=125, max_depth=75
    0.9661419630929642
    0.7643016255435418
    
     n_estimators=125, max_depth=100
    0.9661419630929642
    0.7643016255435418
    
     n_estimators=125, max_depth=125
    0.9661419630929642
    0.7643016255435418


Melhor ajuste para Previsão de Maximal Wind


```python
regr_rf2 = RandomForestRegressor(n_estimators=50, max_depth=50, random_state=0, oob_score=True, bootstrap = True)
regr_rf2.fit(data_train, data_train_mw)
print(regr_rf2.score(data_train, data_train_mw))
print(regr_rf2.score(data_test, data_test_mw))
```

    0.9629951225598822
    0.7593567448937373



```python
fig, ax = plt.subplots(1,2)#, figsize=(16,10))
#fig.suptitle('Velocidade Máxima vs Pressão Mínima (1851-2015)', fontsize=28, y=1.06)

ax[0].scatter(data_train['sst'], data_train_mw, alpha=0.5, label=r'$Dados$ $de$ $Treino$')
ax[0].scatter(data_train['sst'], regr_rf2.predict(data_train), alpha=0.5, label=r'$Previsão$')
ax[1].scatter(data_test['sst'], data_test_mw, alpha=0.5, label=r'$Dados$ $de$ $Teste$')
ax[1].scatter(data_test['sst'], regr_rf2.predict(data_test), alpha=0.5, label=r'$Previsão$')

ax[0].tick_params(labelsize=24)
ax[0].set_title(f'Previsão de Maximal Wind por Random Forest (Treino)', fontsize=24)
ax[0].set_xlabel(r'$sst$', fontsize=16)
ax[0].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[0].legend(loc='best', fontsize=12);

ax[1].tick_params(labelsize=24)
ax[1].set_title(f'Previsão de Maximal Wind por Random Forest (Teste)', fontsize=24)
ax[1].set_xlabel(r'$sst$', fontsize=16)
ax[1].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[1].legend(loc='best', fontsize=12);

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```


![png](Analises_variaveis_files/Analises_variaveis_58_0.png)



```python
fig, ax = plt.subplots(1,2)#, figsize=(16,10))
#fig.suptitle('Velocidade Máxima vs Pressão Mínima (1851-2015)', fontsize=28, y=1.06)
data_concat_train = pd.concat([data_train, data_train_mw], axis=1)
data_concat_test = pd.concat([data_test, data_test_mw], axis=1)

ax[0].scatter(data_concat_train['Year'], data_concat_train['Maximum Wind'], alpha=0.5, label=r'$Dados$ $de$ $Treino$')
ax[0].scatter(data_concat_train['Year'], regr_rf2.predict(data_train), alpha=0.5, label=r'$Previsão$')
ax[1].scatter(data_concat_test['Year'], data_concat_test['Maximum Wind'], alpha=0.5, label=r'$Dados$ $de$ $Teste$')
ax[1].scatter(data_concat_test['Year'], regr_rf2.predict(data_test), alpha=0.5, label=r'$Previsão$')

ax[0].tick_params(labelsize=24)
ax[0].set_title(f'Previsão de Maximal Wind por Random Forest (Treino)', fontsize=24)
ax[0].set_xlabel(r'$Ano$', fontsize=16)
ax[0].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[0].legend(loc='best', fontsize=12);

ax[1].tick_params(labelsize=24)
ax[1].set_title(f'Previsão de Maximal Wind por Random Forest (Teste)', fontsize=24)
ax[1].set_xlabel(r'$Ano$', fontsize=16)
ax[1].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[1].legend(loc='best', fontsize=12);

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```


![png](Analises_variaveis_files/Analises_variaveis_59_0.png)


### Previsão da duração dos eventos de Furacão


```python
data_train2 = data_atl_merged.drop(['ID', 'Name', 'Date', 'Time', 'Event', 'Status', 'Maximum Wind', 'Minimum Pressure', 'Date_c', 'Latitude_c', 'Longitude_c', 'Duration', 'wspd'], 1)
#data_train_mw = data_atl_merged['Maximum Wind']
data_train_dur = data_atl_merged['Duration']
#print(len(data_train))
#data_train.head()
data_train2, data_test2, data_train_dur, data_test_dur = train_test_split(data_train2, data_train_dur, random_state=1)

```

Abaixo, faremos também a previsão da duração de um Furacão. O ajuste fica bem preciso, como se pode ver pelo R2 Score


```python
regr_rf3 = RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf3.fit(data_train2, data_train_dur)
print(regr_rf3.score(data_train2, data_train_dur))
print(regr_rf3.score(data_test2, data_test_dur))
```

    0.9883397102866289
    0.9289716458290775



```python
fig, ax = plt.subplots(1,2)#, figsize=(16,10))
#fig.suptitle('Velocidade Máxima vs Pressão Mínima (1851-2015)', fontsize=28, y=1.06)
data_concat_train = pd.concat([data_train, data_train_mw, data_train_dur], axis=1)
data_concat_test = pd.concat([data_test, data_test_mw, data_test_dur], axis=1)

ax[0].scatter(data_concat_train['Year'], data_concat_train['Duration'], alpha=0.05, label=r'$Dados$ $de$ $Treino$')
ax[0].scatter(data_concat_train['Year'], regr_rf3.predict(data_train2), alpha=0.05, label=r'$Previsão$')
ax[1].scatter(data_concat_test['Year'], data_concat_test['Duration'], alpha=0.05, label=r'$Dados$ $de$ $Teste$')
ax[1].scatter(data_concat_test['Year'], regr_rf3.predict(data_test2), alpha=0.05, label=r'$Previsão$')

ax[0].tick_params(labelsize=24)
ax[0].set_title(f'Previsão da Duração por Random Forest (Treino)', fontsize=24)
ax[0].set_xlabel(r'$Ano$', fontsize=16)
ax[0].set_ylabel(r'$Duração$', fontsize=16)
ax[0].legend(loc='best', fontsize=12);

ax[1].tick_params(labelsize=24)
ax[1].set_title(f'Previsão da Duração por Random Forest (Teste)', fontsize=24)
ax[1].set_xlabel(r'$Ano$', fontsize=16)
ax[1].set_ylabel(r'$Duração$', fontsize=16)
ax[1].legend(loc='best', fontsize=12);

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```


![png](Analises_variaveis_files/Analises_variaveis_64_0.png)


### Multi Layer Perceptron


```python
regr_mlp = MLPRegressor(hidden_layer_sizes=(100,2), random_state=1, max_iter=1000, solver='lbfgs', activation='relu').fit(X_train, y_train_mw)
#regr.predict(X_test[:2])
print(regr_mlp.score(X_train, y_train_mw))
print(regr_mlp.score(X_test, y_test_mw))
```

    /home/gambitura/anaconda3/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:471: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
      self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)


    0.09520915346048042
    0.08725614810614979



```python
fig, ax = plt.subplots(1,2)#, figsize=(16,10))
#fig.suptitle('Velocidade Máxima vs Pressão Mínima (1851-2015)', fontsize=28, y=1.06)

ax[0].scatter(X_train['sst'], y_train_mw, alpha=0.5, label=r'$Dados$ $de$ $Treino$')
ax[0].scatter(X_train['sst'], regr_mlp.predict(X_train), alpha=0.5, label=r'$Previsão$')
ax[1].scatter(X_test['sst'], y_test_mw, alpha=0.5, label=r'$Dados$ $de$ $Teste$')
ax[1].scatter(X_test['sst'], regr_mlp.predict(X_test), alpha=0.5, label=r'$Previsão$')

ax[0].tick_params(labelsize=24)
ax[0].set_title(f'Previsão de Maximal Wind por MLPRegressor (Treino)', fontsize=24)
ax[0].set_xlabel(r'$sst$', fontsize=16)
ax[0].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[0].legend(loc='best', fontsize=12);

ax[1].tick_params(labelsize=24)
ax[1].set_title(f'Previsão de Maximal Wind por MLPRegressor (Teste)', fontsize=24)
ax[1].set_xlabel(r'$sst$', fontsize=16)
ax[1].set_ylabel(r'$Maximal$ $Wind$', fontsize=16)
ax[1].legend(loc='best', fontsize=12);

fig.set_figheight(5)
fig.set_figwidth(20)
fig.tight_layout(pad=2.0)
```


![png](Analises_variaveis_files/Analises_variaveis_67_0.png)


### Support Vector Machine


```python
regr_svr = svm.SVR()
regr_svr.fit(X_train, y_train_mw)
print(regr_svr.score(X_train, y_train_mw))
print(regr_svr.score(X_test, y_test_mw))
```

    -0.06514630260669341
    -0.058505019763586796


### Modelos com Escala Padronizada


```python
# Padronização da Escala
scaler = StandardScaler()  # doctest: +SKIP
scaler.fit(X_train)  # doctest: +SKIP
X_train_std = scaler.transform(X_train)  # doctest: +SKIP
X_test_std = scaler.transform(X_test)
```


```python
regr_svr_std = svm.SVR()
regr_svr_std.fit(X_train_std, y_train_mw)
print(regr_svr_std.score(X_train_std, y_train_mw))
print(regr_svr_std.score(X_test_std, y_test_mw))
```

    0.07772469466765619
    0.07604328404370786



```python
regr_mlp_std = MLPRegressor(hidden_layer_sizes=(100,2), random_state=1, max_iter=1000, solver='lbfgs', activation='relu').fit(X_train, y_train_mw)
#regr.predict(X_test[:2])
print(regr_mlp_std.score(X_train_std, y_train_mw))
print(regr_mlp_std.score(X_test_std, y_test_mw))
```

    /home/gambitura/anaconda3/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:471: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
      self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)


    -4.16270156850149
    -4.153096948726193



```python
regr_rf_std= RandomForestRegressor(n_estimators=50, max_depth=75, random_state=0, oob_score=True, bootstrap = True)
regr_rf_std.fit(X_train_std, y_train_mw)
print(regr_rf_std.score(X_train_std, y_train_mw))
print(regr_rf_std.score(X_test_std, y_test_mw))
```

    0.9298162379440262
    0.5220914993160997

