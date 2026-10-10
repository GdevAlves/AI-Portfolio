# Calibracao do Simulated Annealing
Varredura dos parametros de temperatura do Simulated Annealing contra o Hill Climbing, que serve de referencia fixa. Cada celula traz a media do fitness de 10 execucoes com a mesma semente e, portanto, com os mesmos pontos de partida dos dois algoritmos. O perfil '15 min, so decrescente' nao pede atributos e deixa a progressao dominar o fitness.
| T0 | resfriamento | T min | passos | 30 min, alvos, crescente | 60 min, alvos, pico | 15 min, so decrescente | media geral |
|---|---|---|---|---:|---:|---:|---:|
| 50.0 | 0.95 | 0.01 | 166 | 0.872 | 0.858 | 0.958 | 0.896 |
| 1.0 | 0.95 | 0.01 | 89 | 0.883 | 0.860 | 0.946 | 0.896 |
| 0.5 | 0.95 | 0.01 | 76 | 0.870 | 0.851 | 0.953 | 0.891 |
| 0.2 | 0.95 | 0.01 | 58 | 0.880 | 0.862 | 0.952 | 0.898 |
| 0.1 | 0.95 | 0.01 | 44 | 0.869 | 0.859 | 0.933 | 0.887 |
| 1.0 | 0.99 | 0.001 | 687 | 0.918 | 0.905 | 0.981 | 0.934 |
| 0.5 | 0.99 | 0.001 | 618 | 0.907 | 0.910 | 0.979 | 0.932 |
| 0.2 | 0.99 | 0.001 | 527 | 0.918 | 0.907 | 0.983 | 0.936 |
| 0.1 | 0.99 | 0.001 | 458 | 0.920 | 0.914 | 0.980 | 0.938 |
| hill_climbing (referencia) | | | | 0.903 | 0.903 | 0.965 | 0.924 |

Configuracao escolhida como padrao: T0=0.5, taxa de resfriamento=0.99, temperatura minima=0.001 (618 passos). Media geral 0.932 contra 0.924 do Hill Climbing, ao custo de mais passos por execucao (618 avaliacoes de fitness contra algumas dezenas de passos do Hill Climbing, que avalia 20 vizinhos por passo).