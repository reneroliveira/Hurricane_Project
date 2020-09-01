# Exploração, Limpeza dos dados e Visualizações

Notebook referência: [Data_Cleaning_and_EDA.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/Notebooks/Data_Cleaning_and_EDA.ipynb)

Trabalharemos com **três datasets**. O primeiro é um subconjunto do [ICOADS](https://icoads.noaa.gov/) (International Comprehensive Ocean-Atmosphere Data Set) fornecido pela [NOAA](https://noaa.gov) (National Oceanic and Atmospheric Administration) que possui vários dados climáticos de cobertura mensal, desde 1800 até o presente, e com abrangência marítima; Possui dados como temperatura do mar, umidade, pressão, cobertura de nuvens, velocidade de vento, etc.

Caso queira trabalhar com eles, as instrucões para baixá-los estão no notebook referência.

O segundo grande conjunto de dados é o [HURDAT2](https://www.kaggle.com/noaa/hurricane-database) cuja fonte oficial é a [NHC](www.nhc.noaa.gov) (National Hurricane Center), divisão da NOAA responsável pelos furações e tempestades.

Os dados do Kaggle fornecem dados de tempestades e furacões desde o século XIX até 2015 no pacífico e atlântico, mas iremos focar nossa análise no dadaset do atlântico.

O terceiro, é um dado com o índice PDI, mas ele já vem da fonte em boas condições e não necessita de limpeza. Veja mais no notebook [PowerDissipationIndex.ipynb](https://github.com/reneroliveira/Hurricane_Project/blob/master/PowerDissipationIndex.ipynb).


## Dataset ICOADS

Os dados da ICOADS vem no formato .nc que é legível ao python através da biblioteca netCDF4 que é um dos pré-requisitos para execução dos notebooks.


### Análise de dados faltantes


![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_7_0.png)



__Legenda:__

- sst: Sea Surface Temperature (Temperatura na superfície do mar)
- wspd: Scalar Wind Speed (Velocidade de vento escalar)
- rhum: Relative Humidity (Umidade relativa)
- slp: Sea Level Pressure (Pressão no nível do mar)
- vwnd: V-wind component (Componente V-wind)
- cldc: Cloudiness (Nebulosidade das nuvens)


Como os continentes representam aproximadamente $29,1\%$ da suferfície terrestre e nossos dados só preenchem os oceanos, os continentes são preenchidos como dados inexistentes. Então naturalmente nossa cota inferior de dados faltantes é essa porcentagem.

Os dados lidos dos arquivos .nc vem no formato de "numpy masked array" de 3 dimensões (tempo,latitude,longitude) que tem um atributo "mask" que é indicadora de dado faltante, isso nos ajudará a lidar com esses dados.

Como vemos no plot acima temos várias décadas com níveis de dados faltantes acima de $90\%$ mas vamos analisar focadamente e atlântico norte, que é nossa região de estudos.


![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_9_0.png)


\*MDR é a abreviação de _Main Development Region_ ou região central de desenvolvimento dos furacões no Atlântico Norte e se refere à faixa 10°-20°N, 80°-20°W.

Vemos que a partir de 1950-1960, os dados começam a ficar mais completos na região de estudos. Então, para entender a relação as variáveis, iremos trabalhar a partir desta data.

Entretanto, nada nos impede de usar os dados mais antigos, já que as medições não variam muito quando estão perto, se quisermos trabalhar com tendências de longo prazo podemos cortar os dados a partir de 1920, trabalhar com a média das regiões estudadas, mesmo que com $~70\%$ de dados faltantes. Isso pois temos a array indicadora, que pode ajudar em modelos, e também essa porcentagem é um pouco mais baixa devido às faixas continentais considaradas no corte de coordenadas.



### Visualização

Abaixo temos um exemplo de como os dados de temperatura estão distribuídos em Janeiro de 1955 

![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_13_0.png)


### Agregação dos Dados MDR

Criaremos um dataframe com as médias espaciais da região MDR, para análise futura com o PDI. Usamos essa média baseados na premissa razoável de que nesse corte espacial da MDR do atlântico os valores não variam muito dentro de um mês. Fazemos essa médias para análises de mais longo prazo como podem ver no notebook do [PDI](https://github.com/reneroliveira/Hurricane_Project/blob/master/PowerDissipationIndex.ipynb).

Mais detalhes sobre a criação desse dataframe veja no notebook referência.


Um plot que se tornou possível após essa agregação foi o mostrado abaixo:


![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_16_0.png)


Podemos ver que após 1970-1980 inicia-se uma tendência crescente de aumento de temperatura do mar em relação à média histórica.

Temos também a matriz de correlação entre as variáveis desde novo dataframe:


<style  type="text/css" >
    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col0 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col1 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col2 {
            background-color:  #d6dce4;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col3 {
            background-color:  #b9d0f9;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col4 {
            background-color:  #d6dce4;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col5 {
            background-color:  #f4987a;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col6 {
            background-color:  #c9d7f0;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col7 {
            background-color:  #dfdbd9;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col8 {
            background-color:  #bcd2f7;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col0 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col1 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col2 {
            background-color:  #e36b54;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col3 {
            background-color:  #f7ba9f;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col4 {
            background-color:  #7ea1fa;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col5 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col6 {
            background-color:  #e5d8d1;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col7 {
            background-color:  #f2cbb7;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col8 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col0 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col1 {
            background-color:  #e7745b;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col2 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col3 {
            background-color:  #ee8669;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col4 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col6 {
            background-color:  #f7b194;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col7 {
            background-color:  #f7a688;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col8 {
            background-color:  #a7c5fe;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col1 {
            background-color:  #eed0c0;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col2 {
            background-color:  #f18f71;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col3 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col4 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col5 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col6 {
            background-color:  #d85646;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col7 {
            background-color:  #ead5c9;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col8 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col0 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col1 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col2 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col4 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col5 {
            background-color:  #e1dad6;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col7 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col8 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col0 {
            background-color:  #f7b497;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col1 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col2 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col3 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col4 {
            background-color:  #f4c5ad;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col5 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col6 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col7 {
            background-color:  #b2ccfb;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col8 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col0 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col1 {
            background-color:  #c3d5f4;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col2 {
            background-color:  #f6bda2;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col3 {
            background-color:  #d95847;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col4 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col5 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col6 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col7 {
            background-color:  #dedcdb;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col8 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col0 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col1 {
            background-color:  #edd2c3;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col2 {
            background-color:  #f6a283;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col3 {
            background-color:  #f2c9b4;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col4 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col5 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col6 {
            background-color:  #efcfbf;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col7 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col8 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col0 {
            background-color:  #cbd8ee;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col1 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col2 {
            background-color:  #efcfbf;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col3 {
            background-color:  #cdd9ec;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col4 {
            background-color:  #e2dad5;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col5 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col6 {
            background-color:  #cedaeb;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col7 {
            background-color:  #b5cdfa;
            color:  #000000;
        }    #T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col8 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }</style><table id="T_f64d8f40_eb28_11ea_88a4_cdafc426433b" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Year</th>        <th class="col_heading level0 col1" >Month</th>        <th class="col_heading level0 col2" >sst</th>        <th class="col_heading level0 col3" >rhum</th>        <th class="col_heading level0 col4" >slp</th>        <th class="col_heading level0 col5" >wspd</th>        <th class="col_heading level0 col6" >vwnd</th>        <th class="col_heading level0 col7" >cldc</th>        <th class="col_heading level0 col8" >sst_anomaly</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row0" class="row_heading level0 row0" >Year</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col0" class="data row0 col0" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col1" class="data row0 col1" >-0.010203</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col2" class="data row0 col2" >0.145686</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col3" class="data row0 col3" >-0.098235</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col4" class="data row0 col4" >0.048908</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col5" class="data row0 col5" >0.646453</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col6" class="data row0 col6" >-0.035567</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col7" class="data row0 col7" >0.241832</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow0_col8" class="data row0 col8" >0.379038</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row1" class="row_heading level0 row1" >Month</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col0" class="data row1 col0" >-0.010203</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col1" class="data row1 col1" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col2" class="data row1 col2" >0.762722</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col3" class="data row1 col3" >0.392842</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col4" class="data row1 col4" >-0.434216</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col5" class="data row1 col5" >-0.385048</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col6" class="data row1 col6" >0.151316</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col7" class="data row1 col7" >0.380311</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow1_col8" class="data row1 col8" >-0.017207</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row2" class="row_heading level0 row2" >sst</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col0" class="data row2 col0" >0.145686</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col1" class="data row2 col1" >0.762722</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col2" class="data row2 col2" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col3" class="data row2 col3" >0.635370</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col4" class="data row2 col4" >-0.632031</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col5" class="data row2 col5" >-0.416486</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col6" class="data row2 col6" >0.432498</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col7" class="data row2 col7" >0.559898</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow2_col8" class="data row2 col8" >0.312656</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row3" class="row_heading level0 row3" >rhum</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col0" class="data row3 col0" >-0.098235</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col1" class="data row3 col1" >0.392842</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col2" class="data row3 col2" >0.635370</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col3" class="data row3 col3" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col4" class="data row3 col4" >-0.768182</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col5" class="data row3 col5" >-0.380635</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col6" class="data row3 col6" >0.803748</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col7" class="data row3 col7" >0.305508</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow3_col8" class="data row3 col8" >0.014696</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row4" class="row_heading level0 row4" >slp</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col0" class="data row4 col0" >0.048908</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col1" class="data row4 col1" >-0.434216</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col2" class="data row4 col2" >-0.632031</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col3" class="data row4 col3" >-0.768182</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col4" class="data row4 col4" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col5" class="data row4 col5" >0.317397</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col6" class="data row4 col6" >-0.812037</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col7" class="data row4 col7" >-0.548567</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow4_col8" class="data row4 col8" >0.129542</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row5" class="row_heading level0 row5" >wspd</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col0" class="data row5 col0" >0.646453</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col1" class="data row5 col1" >-0.385048</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col2" class="data row5 col2" >-0.416486</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col3" class="data row5 col3" >-0.380635</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col4" class="data row5 col4" >0.317397</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col5" class="data row5 col5" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col6" class="data row5 col6" >-0.253309</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col7" class="data row5 col7" >0.007404</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow5_col8" class="data row5 col8" >0.040588</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row6" class="row_heading level0 row6" >vwnd</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col0" class="data row6 col0" >-0.035567</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col1" class="data row6 col1" >0.151316</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col2" class="data row6 col2" >0.432498</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col3" class="data row6 col3" >0.803748</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col4" class="data row6 col4" >-0.812037</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col5" class="data row6 col5" >-0.253309</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col6" class="data row6 col6" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col7" class="data row6 col7" >0.236100</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow6_col8" class="data row6 col8" >-0.002599</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row7" class="row_heading level0 row7" >cldc</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col0" class="data row7 col0" >0.241832</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col1" class="data row7 col1" >0.380311</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col2" class="data row7 col2" >0.559898</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col3" class="data row7 col3" >0.305508</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col4" class="data row7 col4" >-0.548567</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col5" class="data row7 col5" >0.007404</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col6" class="data row7 col6" >0.236100</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col7" class="data row7 col7" >1.000000</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow7_col8" class="data row7 col8" >0.019122</td>
            </tr>
            <tr>
                        <th id="T_f64d8f40_eb28_11ea_88a4_cdafc426433blevel0_row8" class="row_heading level0 row8" >sst_anomaly</th>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col0" class="data row8 col0" >0.379038</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col1" class="data row8 col1" >-0.017207</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col2" class="data row8 col2" >0.312656</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col3" class="data row8 col3" >0.014696</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col4" class="data row8 col4" >0.129542</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col5" class="data row8 col5" >0.040588</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col6" class="data row8 col6" >-0.002599</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col7" class="data row8 col7" >0.019122</td>
                        <td id="T_f64d8f40_eb28_11ea_88a4_cdafc426433brow8_col8" class="data row8 col8" >1.000000</td>
            </tr>
    </tbody></table>



Alguma correlações fortes interessantes:
    
- Temperatura do mar (sst) com ano (Year)
- Temperatura do mar (sst) e umidade (rhum)
- Velocidade de vento (wspd) e ano (Year)
- Pressão (slp) e Umidade (rhum)

Análises mais aprofundadas dessas variáveis veremos no notebook de análise do PDI. 



## Dataset HURDAT2 (Hurricane) - Análise e Limpeza

Passemos agora a analisar os dados de tempestades e furacões.



Segue a visualização da **Temporada de Furacões**:


![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_28_0.png)


Período de furacões se concentra entre os meses de julho e novembro.
Isso deve estar relacionado com o período anterior (verão do hemisfério norte, onde acontece o maior número de casos)
O aquecimento das águas está intimamente ligado com a formação das massas de ar que ocasionam os furacões
Isso nos dá uma pista da forma de correlação temporal que devemos buscar para predizer os eventos.

![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_20_0.png)


Veja acima que os picos de temperatura e umidade, coincidem razoavelmente com a temporada de furacões.


![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_22_0.png)


O pico de nebulosidade das nuvens também coincide de forma razoável com a temporada de furacões. Assim como os valores mais baixos de pressão. Essas características estão relacionadas com a formação do evento de tempestade forte ou furacão.



    


Veja abaixo a relação quase linear entre velocidade máxima sustendada de vento e a pressão mínima dos eventos. Essa relação coincide com o que se sabe da formação do evento de furacão.

![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_31_0.png)





Vemos abaixo que o número de registro de furações tem crescido desde 1850, mas isso se deve à maior capacidade de detecção com o passar dos anos.

![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_35_1.png)







Vemos também que a velocidade máxima de vento sustentada pelos furacões reduziu, mas isso também se deve à maior capacidade de registro de eventos de pequeno porte, que acabam pesando a média para baixo. Assim, para não enviesar nossos dados, filtraremos os registros de pequeno porte; Consideramos apenas tempestades cuja duração em dias é maior que 2, e cuja classificação na escala [Saffir-Simpson](https://en.wikipedia.org/wiki/Saffir%E2%80%93Simpson_scale) seja no mínimo Tempestade Tropical.

Para essa classificação, a velocidade máxima sustentada de vento deve ultrapassar 63km/h o que equivale a 34 knots (milhas náuticas).

![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_37_1.png)


Após algumas análises (veja notebook referência), decidimos aplicar um filtro aos dados, para reduzir viés de capacidade de observação, e detecção de tempestades menores.

Os filtros foram:

- Velocidade Máxima Sustentada > 34 milhas náuticas
- Duração > 2 dias
- Furacões a partir de 1950 (quando a capacidade de medição começa a evoluir
- Latitude entre 0° e 25°N (remover viés dos registros extratropicais)

Fazemos também outras limpezas como formatação de entradas de latitude, longitude e outras variáveis que não vieram da forma ideal.



    


## União dos dados via k-NN ponderado

A partir dos dois grandes conjuntos de dados descritos acima criamos um novo dataframe que pega o dataframe HURDAT2 filtrado acima e busca os dados climáticos nos datasets da ICOADS diretamente dos arquivos .nc. Fizemos essa busca via coordenadas e para lidar com dados faltantes implementaremos um k-NN ponderado pelo inverso das distâncias entre as coordenadas originais e o vizinho considerado no algoritmo. 

### Formula da distância

Dados duas coordenadas $(\varphi_1,\lambda_1)$ e ($\varphi_2,\lambda_2)$ em radianos, a [Fórmula Haversine](https://wikimedia.org/api/rest_v1/media/math/render/svg/a65dbbde43ff45bacd2505fcf32b44fc7dcd8cc0) é capaz de calcular a distância real entre esses dois pontos no mapa:

![teste](https://wikimedia.org/api/rest_v1/media/math/render/svg/a65dbbde43ff45bacd2505fcf32b44fc7dcd8cc0)

Onde $r$ é o raio da Terra.

Usando $r$ aproximadamente igual a $6371$ km o valor de $d$ será a distância em km dos dois pontos dados em coordenadas geográficas.

Detalhes sobre a implementação desse cálculo estão no notebook referência.




### k-NN ponderado por inverso da distância

Implementamos várias funcões auxiliares para ajudar nessa tarefa de coletar vizinhos para cálculo de média. Detalhes estão no notebook referência.

Por conta de dados faltantes no ICOADS, nem sempre existirá registros para todas as coordenadas. Assim, buscamos os $k$ vizinhos mais próximos a calculamos a média entre eles, porém a média é ponderada pelo inverso da distância entre o vizinho e a coordenada original (usando a Fórmula Haversive). Sendo assim, pontos mais distantes terão menor peso na média, enquanto pontos mais próximos serão considerados mais importantes.

Por simplicidade, usamos $k=15$.

Após a execução dos algoritmos implementados, salvamos o dataframe gerado em um csv e o usamos para outras análises.


## Outras Visualizações



![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_68_0.png)




![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_69_0.png)


Nos plots abaixo usaremos a biblioteca troPYcal. Internamente ela tem o dataset HURDAT2 atualizado e possui funções de visualização prontas, simples de serem usadas.

Referência: https://tropycal.github.io/tropycal/examples/index.html


![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_71_1.png)



![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_80_0.png)




![png](Data_Cleaning_and_EDA_files/Data_Cleaning_and_EDA_81_0.png)

