# Lab3-Analizator_wynikow: Automatyczna Analiza Danych

Ten raport został wygenerowany automatycznie.

## Etap 1: Eksploracja danych

Dane wczytane z `https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv`:
```
          gender ethnicity      score fcollege mcollege home urban  unemp  wage  distance  tuition  education income region
rownames                                                                                                                   
1           male     other  39.150002      yes       no  yes   yes    6.2  8.09       0.2  0.88915         12   high  other
2         female     other  48.869999       no       no  yes   yes    6.2  8.09       0.2  0.88915         12    low  other
3           male     other  48.740002       no       no  yes   yes    6.2  8.09       0.2  0.88915         12    low  other
4           male      afam  40.400002       no       no  yes   yes    6.2  8.09       0.2  0.88915         12    low  other
5         female     other  40.480000       no       no   no   yes    5.6  8.09       0.4  0.88915         13    low  other
```

Suma brakujących wartości: **0**

Podstawowe statystyki:
```
             score        unemp         wage     distance      tuition    education
count  4739.000000  4739.000000  4739.000000  4739.000000  4739.000000  4739.000000
mean     50.889029     7.597215     9.500506     1.802870     0.814608    13.807765
std       8.701910     2.763581     1.343067     2.297128     0.339504     1.789107
min      28.950001     1.400000     6.590000     0.000000     0.257510    12.000000
25%      43.924999     5.900000     8.850000     0.400000     0.484990    12.000000
50%      51.189999     7.100000     9.680000     1.000000     0.824480    13.000000
75%      57.769999     8.900000    10.150000     2.500000     1.127020    16.000000
max      72.809998    24.900000    12.960000    20.000000     1.404160    18.000000
```

![Dystrybucja Score](plots\score_distribution.png)

![Macierz Korelacji](plots\correlation_matrix.png)

## Wyniki oceny modeli

| Model             |       R² |     MAE |     MSE |
|:------------------|---------:|--------:|--------:|
| Regresja Liniowa  | 0.352337 | 5.75431 | 49.1138 |
| Lasy Losowe       | 0.291963 | 5.85546 | 53.6921 |
| Gradient Boosting | 0.367053 | 5.68634 | 47.9978 |

Najlepszy model pod względem R²: **Gradient Boosting**.
