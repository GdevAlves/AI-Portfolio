# Portfólio — Parte 3: Tratando Incerteza em Inteligência Artificial

---

| Campo | Informação |
|---|---|
| **Autor** | Gabriel Alves |
| **Matrícula** | 241025523 |
| **Disciplina** | FGA0221 – Inteligência Artificial |
| **Professor** | Fabiano Araujo Soares |
| **Turma** | T01 — 2º/2026 |
| **Data de entrega** | 05/11/2026 |

---

## Sumário

1. [Introdução](#1-introdução)
   - 1.1 [Organização do Relatório](#11-organização-do-relatório)
2. [Projetos](#2-projetos)
   - 2.1 [Projeto 1: Redes Bayesianas](#21-projeto-1-redes-bayesianas)
   - 2.2 [Projeto 2: Modelos Markovianos Ocultos (HMM)](#22-projeto-2-modelos-markovianos-ocultos-hmm)
   - 2.3 [Projeto 3: Filtros de Kalman](#23-projeto-3-filtros-de-kalman)
3. [Considerações Finais Gerais](#3-considerações-finais-gerais)
4. [Referências Bibliográficas](#referências-bibliográficas)

---

## 1 Introdução

Este relatório integra a **Parte 3 do Portfólio** da disciplina FGA0221 – Inteligência Artificial (2º/2026), conforme o Plano de Ensino do Prof. Fabiano Araujo Soares. O tema desta parte é **Tratando Incerteza**, cobrindo três técnicas probabilísticas fundamentais: Redes Bayesianas, Modelos Markovianos Ocultos (HMM) e Filtros de Kalman.

Cada um dos três projetos a seguir é acompanhado de: (i) descrição do problema, (ii) descrição detalhada do algoritmo desenvolvido, (iii) resultados com imagens e valores e (iv) considerações finais. Os arquivos Python comentados são entregues em conjunto, nomeados conforme indicado ao início de cada seção.

### 1.1 Organização do Relatório

O relatório está estruturado em três seções principais, uma por projeto:

- **Projeto 1** — Redes Bayesianas &nbsp;&nbsp;`proj1_redes_bayesianas.py`
- **Projeto 2** — Modelos Markovianos Ocultos (HMM) &nbsp;&nbsp;`proj2_hmm.py`
- **Projeto 3** — Filtros de Kalman &nbsp;&nbsp;`proj3_filtro_kalman.py`

---

## 2 Projetos

---

### 2.1 Projeto 1: Redes Bayesianas

> **Arquivo:** `proj1_redes_bayesianas.py`

#### Descrição do Problema

<!-- Descreva o domínio modelado como Rede Bayesiana. Explique as variáveis envolvidas,
     a estrutura do grafo direcional acíclico (DAG) e as relações de dependência condicional
     entre as variáveis. Justifique por que uma Rede Bayesiana é adequada para este problema. -->

#### Algoritmo Desenvolvido

<!-- Descreva a construção da rede (nós, arcos, tabelas de probabilidade condicional — CPTs)
     e o algoritmo de inferência utilizado (ex.: eliminação de variável, Belief Propagation).
     Explique como consultas de probabilidade são respondidas e quais bibliotecas Python foram
     utilizadas (ex.: pgmpy, pomegranate). -->

#### Resultados

<!-- Mostre a estrutura do grafo (imagem ou diagrama), as tabelas de probabilidade condicional
     (CPTs) e os resultados de consultas específicas realizadas sobre a rede. Apresente
     comparações entre probabilidades a priori e a posteriori dadas as evidências observadas. -->

#### Considerações Finais

<!-- Discuta a complexidade da inferência exata vs. aproximada, limitações da estrutura do
     grafo escolhido, sensibilidade à especificação das CPTs e extensões possíveis
     (ex.: Redes Bayesianas Dinâmicas). -->

---

### 2.2 Projeto 2: Modelos Markovianos Ocultos (HMM)

> **Arquivo:** `proj2_hmm.py`

#### Descrição do Problema

<!-- Descreva o problema modelado como HMM. Identifique os estados ocultos, as observações
     possíveis e o contexto de aplicação (ex.: reconhecimento de padrões, análise de séries
     temporais, detecção de anomalias). Explique por que a estrutura Markoviana é adequada
     para o domínio escolhido. -->

#### Algoritmo Desenvolvido

<!-- Descreva os componentes do HMM implementado: matriz de transição entre estados (A),
     matriz de emissão/observação (B) e distribuição inicial de estados (π). Explique os
     algoritmos utilizados: Forward-Backward para inferência, Viterbi para decodificação
     do estado mais provável e/ou Baum-Welch para aprendizado dos parâmetros. -->

#### Resultados

<!-- Apresente as matrizes A, B e π obtidas (após treinamento ou definidas manualmente),
     a sequência de estados ocultos mais provável para uma sequência de observações de teste
     (resultado do Viterbi) e métricas de verossimilhança (log-likelihood) do modelo. -->

#### Considerações Finais

<!-- Discuta a hipótese de Markov de primeira ordem (limitação de memória), a sensibilidade
     à inicialização dos parâmetros no Baum-Welch, a escolha do número de estados e
     possíveis extensões (HMMs de alta ordem, modelos de mistura). -->

---

### 2.3 Projeto 3: Filtros de Kalman

> **Arquivo:** `proj3_filtro_kalman.py`

#### Descrição do Problema

<!-- Descreva o problema de estimativa de estado dinâmico sob ruído. Defina as variáveis de
     estado, as observações disponíveis e as fontes de incerteza (ruído de processo e ruído
     de medição). Explique por que o Filtro de Kalman é adequado para este sistema. -->

#### Algoritmo Desenvolvido

<!-- Descreva as matrizes do modelo: F (transição de estado), H (observação), Q (covariância
     do ruído de processo), R (covariância do ruído de medição) e P (covariância do erro de
     estimativa). Explique as duas etapas do filtro: Predição (estado e covariância a priori)
     e Atualização (ganho de Kalman, correção do estado e covariância a posteriori). -->

#### Resultados

<!-- Apresente gráficos comparando o estado estimado pelo filtro vs. o estado real (se
     disponível) vs. as observações ruidosas. Inclua métricas de erro (MSE, RMSE) e
     visualização da incerteza (intervalo de confiança ±2σ ao longo do tempo). -->

#### Considerações Finais

<!-- Discuta as hipóteses de linearidade e gaussianidade do modelo, os cenários em que elas
     se quebram e alternativas mais robustas (Filtro de Kalman Estendido — EKF,
     Filtro de Kalman Unscented — UKF, Filtro de Partículas). -->

---

## 3 Considerações Finais Gerais

<!-- Redija uma síntese comparativa dos três projetos: em quais tipos de problema cada técnica
     se mostrou mais adequada, o que Redes Bayesianas, HMMs e Filtros de Kalman têm em comum
     (modelagem probabilística explícita da incerteza) e em que se diferenciam (estrutura
     estática vs. dinâmica, inferência exata vs. aproximada, domínio discreto vs. contínuo).
     Reflita sobre a relevância dessas abordagens frente aos métodos clássicos da Parte 2
     e às técnicas de aprendizado de máquina modernas. -->

---

## Referências Bibliográficas

RUSSELL, S.; NORVIG, P. **Artificial Intelligence: A Modern Approach**. 4. ed. Pearson, 2022.

BISHOP, C. M. **Pattern Recognition and Machine Learning**. Springer, 2006.

WELCH, G.; BISHOP, G. **An Introduction to the Kalman Filter**. University of North Carolina at Chapel Hill, 2006. Disponível em: https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf.

<!-- Acrescente aqui as demais referências utilizadas, seguindo as normas ABNT. -->