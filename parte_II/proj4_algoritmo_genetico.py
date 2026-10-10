#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 4 - ALGORITMO GENETICO
==============================
Geracao de playlists com um algoritmo genetico sobre o mesmo catalogo de cem
faixas reais do Spotify usado no projeto 3 (busca em ambientes complexos).
A reutilizacao deliberada do problema permite comparar os dois paradigmas
(busca local contra evolucao de populacao) sobre exatamente a mesma funcao de
qualidade.

PROBLEMA FORMAL (Russell & Norvig, AIMA 4a ed., capitulo 4)
  - Agente      : montador de playlists com acesso a um catalogo de 100 faixas
  - Estados     : permutacoes de N faixas distintas do catalogo (individuos)
  - Acoes       : os operadores geneticos (selecao, cruzamento e mutacao)
  - Custo c(n,m): nao ha custo de caminho; cada estado e uma solucao completa
                  avaliada diretamente pela funcao de fitness
  - Objetivo    : maximizar o fitness ponderado (compatibilidade, duracao,
                  progressao, transicoes e diversidade)
  - Restricao   : "mole". A duracao alvo e perseguida pela nota de duracao, e
                  nao por corte exato; o tamanho N do individuo e fixo por
                  perfil e as faixas nunca se repetem dentro de um individuo.

O que torna o problema adequado a um algoritmo genetico e a mesma estrutura
combinatoria do projeto 3: o espaco de candidatos e astronomico e a funcao de
qualidade tem muitos otimos locais. Onde o Hill Climbing sobe um unico pico e
o Simulated Annealing aceita pioras para escapar dele, o algoritmo genetico
mantem uma populacao inteira de playlists, combina pedacos de boas solucoes
pelo cruzamento e explora o espaco pela mutacao, o que troca otimos locais
isolados por uma cobertura ampla da fronteira de boas solucoes.

REPRESENTACAO
  Cada individuo e uma playlist de comprimento fixo N, estimado a partir da
  duracao alvo e da duracao media do catalogo (teto de 30 faixas). Cada gene
  guarda uma faixa distinta, e a posicao do gene importa: a ordem das faixas
  define a progressao de energia e as transicoes. Com comprimento fixo, o
  cruzamento Order Crossover (OX) funciona sem criar faixas duplicadas.

OPERADORES
  - Selecao  : torneio de k individuos, vence o de maior fitness
  - Cruzamento: Order Crossover (OX) com dois pontos de corte
  - Mutacao  : um dos tres movimentos do projeto 3 (troca de posicoes,
               substituicao por faixa fora do individuo, inversao de trecho)
  - Elitismo : os melhores individuos passam intactos para a proxima geracao

EXPERIMENTOS
  1. Convergencia do melhor e da media de fitness por geracao (3 perfis)
  2. Varredura do tamanho da populacao  (20, 50, 100, 200)
  3. Varredura da taxa de mutacao       (1%, 5%, 10%, 20%)
  4. Comparacao com os baselines aleatorio e guloso do projeto 3, cujos
     numeros de Hill Climbing e Simulated Annealing sao citados no relatorio.

DEPENDENCIAS
  matplotlib (figuras). Nada alem da biblioteca padrao.

EXECUCAO
  python proj4_algoritmo_genetico.py
  Os perfis do experimento estao fixados como constantes para tornar as
  varreduras reproduziveis. Para reproduzir os numeros exatos do relatorio,
  rode com PYTHONHASHSEED=0: a ordem de iteracao dos conjuntos do Python
  depende do hash de strings, aleatorio entre processos, e o movimento de
  substituicao sorteia a partir de um conjunto (mesma ressalva do projeto 3).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURACAO
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DADOS = os.path.join(BASE, "data")
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

# Caminho do dataset. O arquivo e o mesmo do projeto 3, gerado uma unica vez
# a partir da API real do Spotify e da API FreqBlog, e depois apenas lido. Se
# o arquivo nao existir, o programa avisa e encerra, sem acessar a internet.
ARQUIVO_DATASET = os.path.join(DIR_DADOS, "spotify_tracks.json")

# Link publico do dataset na nuvem, para o professor baixar o arquivo caso a
# copia local nao acompanhe a entrega.
URL_DATASET_NUVEM = ""

# Semente para tornar o experimento reproduzivel.
SEMENTE = 42

# Quantas vezes cada configuracao e executada no experimento final.
N_EXECUCOES = 5

# Pesos de cada componente da funcao de qualidade, herdados do projeto 3 para
# manter a comparacao entre os paradigmas valida. A soma e exatamente 1,0.
PESO_COMPATIBILIDADE = 0.30
PESO_DURACAO = 0.20
PESO_PROGRESSAO = 0.25
PESO_TRANSICOES = 0.15
PESO_DIVERSIDADE = 0.10

# Parametros padrao do algoritmo genetico.
POPULACAO = 100
TAXA_CROSSOVER = 0.90
TAXA_MUTACAO = 0.10
ELITE = 2
N_GERACOES = 100
ESTAGNACAO = 30
TORNEIO_K = 3

# Tamanho maximo de uma playlist, medido em numero de faixas.
TETO_FAIXAS = 30

# Padroes de progressao de energia aceitos.
PROGRESSOES = ("constante", "crescente", "decrescente", "pico")

# Valores varridos nos experimentos de parametros.
VARREDURA_POPULACAO = (20, 50, 100, 200)
VARREDURA_MUTACAO = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class Faixa:
    """Uma faixa do catalogo com os atributos numericos usados na busca."""

    id: str
    nome: str
    artistas: tuple[str, ...]
    album: str
    ano: int
    duracao_s: float
    explicita: bool
    isrc: str
    url: str
    energia: float
    valencia: float
    acustica: float
    dancabilidade: float
    andamento: float
    tom: int
    modo: int
    volume_db: float


@dataclass(frozen=True)
class Perfil:
    """O que o usuario pediu: alvos numericos e padrao de progressao.

    Qualquer alvo pode ser None quando o usuario nao se importa com aquele
    atributo. A duracao alvo nunca e None porque o tamanho do individuo e
    estimado a partir dela.
    """

    duracao_alvo_s: float
    energia: float | None
    valencia: float | None
    acustica: float | None
    dancabilidade: float | None
    andamento: float | None
    progressao: str


# ---------------------------------------------------------------------------
# 2. CARREGAMENTO DO DATASET
# ---------------------------------------------------------------------------


def carregar_catalogo(caminho: str = ARQUIVO_DATASET) -> list[Faixa]:
    """Le o arquivo JSON com as cem faixas e devolve uma lista de Faixa.

    O arquivo foi produzido uma unica vez com dados reais (Spotify para as
    faixas, FreqBlog para as caracteristicas numericas) e este programa nunca
    tenta baixa-lo de novo. Se o arquivo nao existir, a mensagem de erro
    indica o link da nuvem onde a copia do dataset esta disponivel.
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            "Dataset nao encontrado em " + caminho + ". Baixe o arquivo "
            "spotify_tracks.json do link " + (URL_DATASET_NUVEM or "indicado no relatorio")
            + " e coloque-o na pasta data/."
        )

    with open(caminho, encoding="utf-8") as fh:
        bruto = json.load(fh)

    catalogo: list[Faixa] = []
    for item in bruto["faixas"]:
        catalogo.append(Faixa(
            id=item["id"],
            nome=item["nome"],
            artistas=tuple(item["artistas"]),
            album=item["album"],
            ano=int(item["ano_lancamento"]),
            duracao_s=item["duracao_ms"] / 1000.0,
            explicita=item["explicit"],
            isrc=item["isrc"],
            url=item["url_spotify"],
            energia=item["energy"],
            valencia=item["valence"],
            acustica=item["acousticness"],
            dancabilidade=item["danceability"],
            andamento=item["tempo"],
            tom=item["key"],
            modo=item["mode"],
            volume_db=item["loudness"],
        ))

    if len(catalogo) != 100:
        print("Aviso: o dataset tem", len(catalogo), "faixas, esperava-se 100.")

    return catalogo


def _total_playlist(playlist: list[Faixa]) -> float:
    """Duracao total em segundos de uma lista de faixas."""
    return sum(f.duracao_s for f in playlist)


def formatar_tempo(segundos: float) -> str:
    """Converte segundos em texto mm:ss, como aparece nas interfaces de musica."""
    minutos = int(segundos // 60)
    restante = int(round(segundos % 60))
    if restante == 60:
        minutos += 1
        restante = 0
    return f"{minutos:02d}:{restante:02d}"


def _barra_energia(valores: list[float]) -> str:
    """Monta a linha de progressao de energia impressa no final."""
    if not valores:
        return ""
    largura = 40
    minimo, maximo = min(valores), max(valores)
    faixa = max(maximo - minimo, 1e-9)
    linha = ""
    for valor in valores:
        pos = int((valor - minimo) / faixa * (largura - 1))
        linha += " " * pos + "*\n"
    return linha.rstrip("\n")


def imprimir_playlist(playlist: list[Faixa], fitness: float, perfil: Perfil) -> None:
    """Imprime a playlist final no formato do planejamento do projeto.

    Mostra as faixas numeradas com artista, nome, album e duracao, o total
    de minutos, o fitness e a curva de energia da playlist, que permite
    conferir visualmente se a progressao pedida foi atendida.
    """
    print()
    print("=" * 46)
    print("                PLAYLIST ENCONTRADA")
    print("=" * 46)
    print()
    for i, faixa in enumerate(playlist, start=1):
        artistas = ", ".join(faixa.artistas)
        print(f"{i:02d}. {artistas} - {faixa.nome}")
        print(f"    {faixa.album} ({faixa.ano})  {formatar_tempo(faixa.duracao_s)}")
    print()
    print(f"Duracao:     {formatar_tempo(_total_playlist(playlist))}")
    print(f"Fitness:     {fitness:.3f}")
    print(f"Progressao:  {perfil.progressao}")
    print()
    print("Energia ao longo da playlist:")
    print(_barra_energia([f.energia for f in playlist]))
    print()


# ---------------------------------------------------------------------------
# 3. FUNCAO DE FITNESS
# ---------------------------------------------------------------------------
# Copiada do projeto 3 sem alteracoes, para que a comparacao entre busca
# local e algoritmo genetico seja feita sobre exatamente a mesma medida.
# Cada componente devolve um numero entre 0 e 1. O fitness final e a soma
# ponderada pelos pesos da secao de configuracao.


def compatibilidade_perfil(faixa: Faixa, perfil: Perfil) -> float:
    """Nota de uma faixa isolada, medindo o quanto ela se aproxima do perfil.

    Para cada atributo pedido no perfil, calcula a distancia absoluta entre
    o valor da faixa e o alvo e tira a media das distancias. O andamento e
    medido em BPM e entra dividido por uma faixa de referencia de 100, para
    ficar na mesma escala dos demais atributos. A nota final e 1 menos a
    distancia media, limitada entre 0 e 1. Perfil sem nenhum atributo pedido
    devolve 1.0, pois qualquer faixa atende.
    """
    if perfil.energia is None and perfil.valencia is None and \
       perfil.acustica is None and perfil.dancabilidade is None and \
       perfil.andamento is None:
        return 1.0

    distancias = []
    for atributo, valor_faixa, valor_perfil in [
        ("energia", faixa.energia, perfil.energia),
        ("valencia", faixa.valencia, perfil.valencia),
        ("acustica", faixa.acustica, perfil.acustica),
        ("dancabilidade", faixa.dancabilidade, perfil.dancabilidade),
        ("andamento", faixa.andamento, perfil.andamento),
    ]:
        if valor_perfil is not None:
            distancias.append(abs(valor_faixa - valor_perfil))
            if atributo == "andamento":
                distancias[-1] /= 100.0

    media_distancia = sum(distancias) / len(distancias)
    nota = 1.0 - media_distancia
    return max(0.0, min(1.0, nota))  # garante que a nota fique entre 0 e 1


def pontuacao_duracao(playlist: list[Faixa], perfil: Perfil) -> float:
    """Nota da duracao total, comparada com o alvo pedido no perfil.

    O erro absoluto em segundos entre a soma das duracoes e o alvo decai
    exponencialmente com tau de 180 s, o que da nota proxima de 1 para
    desvios pequenos e nota baixa para desvios grandes.
    """
    duracao_total = sum(f.duracao_s for f in playlist)
    erro = abs(duracao_total - perfil.duracao_alvo_s)
    nota = math.exp(-erro / 180.0)
    return max(0.0, min(1.0, nota))


def _curva_desejada(n_faixas: int, progressao: str) -> list[float]:
    """Devolve a curva de energia descrita pelo padrao de progressao.

    A curva tem um valor por posicao da playlist. As posicoes sao mapeadas
    para o intervalo entre 0 e 1 (primeira faixa em 0, ultima em 1) e cada
    padrao aplica sua regra sobre essa posicao. Crescente usa a propria
    posicao, decrescente usa 1 menos a posicao, pico sobe ate a metade e
    desce depois, e constante devolve sempre 0.5.
    """
    if n_faixas <= 1:
        return [0.5] * max(n_faixas, 0)

    if progressao == "crescente":
        return [i / (n_faixas - 1) for i in range(n_faixas)]
    if progressao == "decrescente":
        return [1 - i / (n_faixas - 1) for i in range(n_faixas)]
    if progressao == "pico":
        return [1 - 2 * abs(i / (n_faixas - 1) - 0.5) for i in range(n_faixas)]
    return [0.5] * n_faixas


def pontuacao_progressao(playlist: list[Faixa], perfil: Perfil) -> float:
    """Nota do formato da curva de energia, ignorando o nivel medio.

    Compara a sequencia real de energias com a curva descrita pelo padrao de
    progressao. As duas sequencias sao normalizadas para o intervalo entre 0
    e 1 antes da comparacao, para que o deslocamento vertical nao influencie
    a nota. A distancia e o erro medio quadratico entre as sequencias
    normalizadas e a nota final e 1 menos esse erro, limitada entre 0 e 1.
    Sequencias quase constantes nao tem forma discernivel: para o padrao
    constante recebem nota maxima e para os demais padroes recebem nota
    neutra.
    """
    energias = [f.energia for f in playlist]
    if not energias:
        return 0.5

    curva = _curva_desejada(len(energias), perfil.progressao)
    if not curva:
        return 0.5

    faixa_energias = max(energias) - min(energias)
    faixa_curva = max(curva) - min(curva)

    # Sem variacao de energia, a forma pedida nao aparece na playlist.
    if faixa_energias < 0.02:
        return 1.0 if perfil.progressao == "constante" else 0.5

    if faixa_curva > 0:
        curva_norm = [(x - min(curva)) / faixa_curva for x in curva]
    else:
        curva_norm = [0.5] * len(curva)

    energias_norm = [(x - min(energias)) / faixa_energias for x in energias]

    erro_quadratico = sum(
        (a - b) ** 2 for a, b in zip(curva_norm, energias_norm)
    ) / len(curva_norm)
    nota = 1 - erro_quadratico

    return max(0.0, min(1.0, nota))


def pontuacao_transicoes(playlist: list[Faixa]) -> float:
    """Nota de suavidade dos saltos de energia entre faixas vizinhas.

    Cada par de faixas consecutivas contribui com 1 menos o modulo da
    diferenca de energia entre elas, e a nota e a media dessas contribuicoes,
    limitada entre 0 e 1. Playlists com menos de duas faixas nao tem
    transicoes e recebem nota maxima.
    """
    if len(playlist) < 2:
        return 1.0

    contribuicoes = [
        1.0 - abs(playlist[i].energia - playlist[i + 1].energia)
        for i in range(len(playlist) - 1)
    ]
    nota = sum(contribuicoes) / len(contribuicoes)
    return max(0.0, min(1.0, nota))


def pontuacao_diversidade(playlist: list[Faixa]) -> float:
    """Nota de variedade de artistas na playlist.

    A base da nota e a proporcao entre artistas distintos e o numero de
    faixas, e cada par de faixas vizinhas com algum artista em comum desconta
    0.1. A nota final e limitada entre 0 e 1.
    """
    if len(playlist) == 0:
        return 1.0

    artistas_distintos = len({a for f in playlist for a in f.artistas})
    nota = artistas_distintos / len(playlist)

    for i in range(len(playlist) - 1):
        atuais = set(playlist[i].artistas)
        proximos = set(playlist[i + 1].artistas)
        if atuais & proximos:
            nota -= 0.1

    return max(0.0, min(1.0, nota))


def fitness(playlist: list[Faixa], perfil: Perfil) -> float:
    """Soma ponderada das cinco notas, o numero que a evolucao maximiza.

    A compatibilidade entra como a media das notas de todas as faixas da
    playlist. Playlist vazia devolve 0.
    """
    if len(playlist) == 0:
        return 0

    # Media de compatibilidade das faixas da playlist.
    compatibilidade = sum(
        compatibilidade_perfil(faixa, perfil) for faixa in playlist
    ) / len(playlist)
    duracao = pontuacao_duracao(playlist, perfil)
    progressao = pontuacao_progressao(playlist, perfil)
    transicoes = pontuacao_transicoes(playlist)
    diversidade = pontuacao_diversidade(playlist)

    fitness_valor = (
        PESO_COMPATIBILIDADE * compatibilidade
        + PESO_DURACAO * duracao
        + PESO_PROGRESSAO * progressao
        + PESO_TRANSICOES * transicoes
        + PESO_DIVERSIDADE * diversidade
    )

    return fitness_valor


# ---------------------------------------------------------------------------
# 4. PERFIS DE EXPERIMENTO
# ---------------------------------------------------------------------------
# Os tres perfis repetem a estrutura dos perfis do projeto 3 (mesmas duracoes
# alvo e mesmos padroes de progressao), para que as tabelas dos dois
# projetos sejam comparaveis. Ficam fixados como constantes porque as
# varreduras de parametros exigem execucoes reproduziveis.

PERFIS_EXPERIMENTO: dict[str, Perfil] = {
    "30 min, alvos, crescente": Perfil(
        duracao_alvo_s=30 * 60.0,
        energia=0.70,
        valencia=0.60,
        acustica=0.30,
        dancabilidade=0.70,
        andamento=120.0,
        progressao="crescente",
    ),
    "60 min, alvos, pico": Perfil(
        duracao_alvo_s=60 * 60.0,
        energia=0.80,
        valencia=0.50,
        acustica=0.20,
        dancabilidade=0.60,
        andamento=110.0,
        progressao="pico",
    ),
    "15 min, so decrescente": Perfil(
        duracao_alvo_s=15 * 60.0,
        energia=None,
        valencia=None,
        acustica=None,
        dancabilidade=None,
        andamento=None,
        progressao="decrescente",
    ),
}


# ---------------------------------------------------------------------------
# 5. CROMOSSOMO E POPULACAO INICIAL
# ---------------------------------------------------------------------------
# O cromossomo e uma lista de N faixas distintas, com a posicao de cada gene
# significativa: a ordem das faixas define progressao e transicoes. O tamanho
# N e fixo por perfil, o que permite o cruzamento OX sem criar faixas
# repetidas, uma exigencia da representacao por permutacao.


def n_faixas_por_perfil(catalogo: list[Faixa], perfil: Perfil) -> int:
    """Estima o tamanho N do cromossomo a partir da duracao alvo.

    Divide a duracao alvo pela duracao media do catalogo e arredonda para
    cima, respeitando o teto de faixas e o tamanho do proprio catalogo, com
    minimo de duas faixas para que progressao e transicoes existam.
    """
    media_s = sum(f.duracao_s for f in catalogo) / len(catalogo)
    estimado = math.ceil(perfil.duracao_alvo_s / media_s)
    return max(2, min(estimado, TETO_FAIXAS, len(catalogo)))


def individuo_aleatorio(catalogo: list[Faixa], n: int) -> list[Faixa]:
    """Sorteia um individuo: n faixas distintas em ordem aleatoria."""
    return random.sample(catalogo, n)


def gerar_populacao(catalogo: list[Faixa], n: int, tamanho: int) -> list[list[Faixa]]:
    """Cria a populacao inicial, toda ela de individuos aleatorios."""
    return [individuo_aleatorio(catalogo, n) for _ in range(tamanho)]


# ---------------------------------------------------------------------------
# 6. OPERADORES GENETICOS
# ---------------------------------------------------------------------------


def selecao_torneio(populacao: list[list[Faixa]], notas: list[float],
                    k: int) -> list[Faixa]:
    """Seleciona um individuo por torneio de k candidatos sorteados.

    O torneio compara uma amostra de k individuos e devolve uma copia do
    vencedor, o de maior fitness. Amostras pequenas deixam a pressao seletiva
    baixa (mais exploracao), amostras grandes aumentam a pressao (mais
    explotacao). A copia evita que os descendentes compartilhem a mesma
    lista com os pais.
    """
    indices = random.sample(range(len(populacao)), min(k, len(populacao)))
    melhor_indice = max(indices, key=lambda i: notas[i])
    return list(populacao[melhor_indice])


def crossover_ox(pai1: list[Faixa], pai2: list[Faixa]) -> list[Faixa]:
    """Cruza dois pais pelo Order Crossover (OX), com dois pontos de corte.

    O OX e o cruzamento classico para permutacoes: preserva um trecho inteiro
    do primeiro pai e completa o filho com os genes restantes na ordem em que
    aparecem no segundo pai, percorrido circularmente a partir do segundo
    corte. O resultado e sempre uma permutacao valida, sem faixa repetida nem
    faixa perdida, algo que um cruzamento de um ponto comum nao garantiria.
    """
    n = len(pai1)
    if n < 2:
        return list(pai1)

    cortes = sorted(random.sample(range(n + 1), 2))
    inicio, fim = cortes

    filho: list[Faixa | None] = [None] * n
    filho[inicio:fim] = pai1[inicio:fim]

    posicao = fim
    for gene in pai2[fim:] + pai2[:fim]:
        if gene not in filho:
            filho[posicao % n] = gene
            posicao += 1

    return [f for f in filho if f is not None]


def mutacao(individuo: list[Faixa], catalogo: list[Faixa]) -> list[Faixa]:
    """Aplica um dos tres movimentos do projeto 3 a um individuo.

    A troca inverte duas posicoes, a substituicao troca uma faixa por outra
    do catalogo que ainda nao esteja no individuo e a inversao inverte todo o
    trecho entre dois cortes. O mutante e sempre uma copia nova da lista,
    nunca a mesma referencia. Os mesmos movimentos compunham a vizinhanca das
    buscas locais do projeto 3; aqui viram a mutacao do algoritmo genetico.
    """
    mutante = list(individuo)
    disponiveis = set(catalogo) - set(individuo)

    movimento = random.choice(["swap", "replace", "invert"])
    if len(mutante) < 2:
        movimento = "replace"

    if movimento == "swap":
        posicoes = random.sample(range(len(mutante)), 2)
        mutante[posicoes[0]], mutante[posicoes[1]] = (
            mutante[posicoes[1]], mutante[posicoes[0]])
    elif movimento == "replace":
        if disponiveis:
            posicao = random.randrange(len(mutante))
            mutante[posicao] = random.choice(list(disponiveis))
    else:
        cortes = sorted(random.sample(range(len(mutante) + 1), 2))
        mutante[cortes[0]:cortes[1]] = reversed(mutante[cortes[0]:cortes[1]])

    return mutante


# ---------------------------------------------------------------------------
# 7. ALGORITMO GENETICO
# ---------------------------------------------------------------------------


def algoritmo_genetico(catalogo: list[Faixa], perfil: Perfil,
                       tam_populacao: int = POPULACAO,
                       taxa_crossover: float = TAXA_CROSSOVER,
                       taxa_mutacao: float = TAXA_MUTACAO,
                       elite: int = ELITE,
                       max_geracoes: int = N_GERACOES,
                       estagnacao: int = ESTAGNACAO,
                       torneio_k: int = TORNEIO_K
                       ) -> tuple[list[Faixa], float, list[float], list[float]]:
    """Evolui uma populacao de playlists e devolve a melhor encontrada.

    A cada geracao, os melhores individuos sao preservados por elitismo e o
    restante da populacao e preenchido por selecao de torneio, cruzamento OX
    (com probabilidade taxa_crossover; do contrario o filho e um clone de um
    dos pais) e mutacao (com probabilidade taxa_mutacao). O laco para quando
    esgota o orcamento de geracoes ou quando o melhor fitness nao melhora por
    estagnacao geracoes seguidas. Devolve a melhor playlist, o fitness dela e
    os historicos do melhor e da media de fitness por geracao, usados na
    figura de convergencia.
    """
    n = n_faixas_por_perfil(catalogo, perfil)
    populacao = gerar_populacao(catalogo, n, tam_populacao)
    notas = [fitness(individuo, perfil) for individuo in populacao]

    melhor_fitness = max(notas)
    melhor = list(populacao[notas.index(melhor_fitness)])
    historico_melhor = [melhor_fitness]
    historico_media = [sum(notas) / len(notas)]
    geracoes_sem_melhora = 0

    for _ in range(1, max_geracoes + 1):
        ordenados = sorted(zip(notas, populacao),
                           key=lambda par: par[0], reverse=True)
        nova_populacao = [list(individuo) for _, individuo in ordenados[:elite]]

        while len(nova_populacao) < tam_populacao:
            pai1 = selecao_torneio(populacao, notas, torneio_k)
            pai2 = selecao_torneio(populacao, notas, torneio_k)
            if random.random() < taxa_crossover:
                filho = crossover_ox(pai1, pai2)
            else:
                filho = list(pai1)
            if random.random() < taxa_mutacao:
                filho = mutacao(filho, catalogo)
            nova_populacao.append(filho)

        populacao = nova_populacao
        notas = [fitness(individuo, perfil) for individuo in populacao]

        melhor_geracao = max(notas)
        historico_melhor.append(melhor_geracao)
        historico_media.append(sum(notas) / len(notas))

        if melhor_geracao > melhor_fitness:
            melhor_fitness = melhor_geracao
            melhor = list(populacao[notas.index(melhor_geracao)])
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1
            if geracoes_sem_melhora >= estagnacao:
                break

    return melhor, melhor_fitness, historico_melhor, historico_media


# ---------------------------------------------------------------------------
# 8. BASELINES
# ---------------------------------------------------------------------------


def playlist_aleatoria(catalogo: list[Faixa], perfil: Perfil) -> list[Faixa]:
    """Playlist de referencia para o pior caso do experimento.

    Embaralha o catalogo e acrescenta faixas enquanto a proxima nao estourar
    a duracao alvo, respeitando o teto de faixas da configuracao.
    """
    faixas = list(catalogo)
    random.shuffle(faixas)
    playlist: list[Faixa] = []
    for faixa in faixas:
        if len(playlist) >= TETO_FAIXAS:
            break
        if _total_playlist(playlist) + faixa.duracao_s <= perfil.duracao_alvo_s:
            playlist.append(faixa)
    return playlist


def playlist_gulosa(catalogo: list[Faixa], perfil: Perfil) -> list[Faixa]:
    """Baseline guloso do experimento.

    Escolhe sempre a faixa restante de maior compatibilidade com o perfil,
    ate que a proxima estoure a duracao alvo. Como o guloso nao decide a
    ordem, a playlist final e ordenada por energia no sentido do padrao de
    progressao pedido, o que deixa a comparacao com o algoritmo genetico
    mais justa.
    """
    playlist: list[Faixa] = []
    disponiveis = set(catalogo)
    while disponiveis:
        melhor = max(disponiveis, key=lambda f: compatibilidade_perfil(f, perfil))
        if _total_playlist(playlist) + melhor.duracao_s > perfil.duracao_alvo_s:
            break
        playlist.append(melhor)
        disponiveis.remove(melhor)

    decrescente = perfil.progressao == "decrescente"
    playlist.sort(key=lambda f: f.energia, reverse=decrescente)
    return playlist


# ---------------------------------------------------------------------------
# 9. EXPERIMENTO
# ---------------------------------------------------------------------------


def _rodar_ga(nome: str, catalogo: list[Faixa], perfil: Perfil,
              **parametros) -> dict:
    """Roda o algoritmo genetico varias vezes e resume media, desvio e melhor.

    Guarda tambem o historico da melhor execucao, para a figura de
    convergencia. Os parametros extras substituem os padroes do algoritmo,
    o que permite reusar a funcao nas varreduras de populacao e de mutacao.
    """
    valores: list[float] = []
    melhor_playlist: list[Faixa] = []
    melhor_fitness = -math.inf
    historico_melhor: list[float] = []
    historico_media: list[float] = []
    for _ in range(N_EXECUCOES):
        t0 = time.perf_counter()
        playlist, valor, hist_melhor, hist_media = algoritmo_genetico(
            catalogo, perfil, **parametros)
        t1 = time.perf_counter()
        valores.append(valor)
        if valor > melhor_fitness:
            melhor_fitness = valor
            melhor_playlist = playlist
            historico_melhor = hist_melhor
            historico_media = hist_media
        print(f"  [{nome}] fitness={valor:.3f}  ({t1 - t0:.2f}s)")

    return {
        "algoritmo": nome,
        "media": statistics.mean(valores),
        "desvio": statistics.pstdev(valores),
        "min": min(valores),
        "max": max(valores),
        "execucoes": N_EXECUCOES,
        "melhor_playlist": [f.nome for f in melhor_playlist],
        "melhor_fitness": melhor_fitness,
        "historico_melhor": historico_melhor,
        "historico_media": historico_media,
    }


def _rodar_baseline(nome: str, funcao, catalogo: list[Faixa],
                    perfil: Perfil) -> dict:
    """Roda um baseline deterministico varias vezes e resume as estatisticas."""
    valores: list[float] = []
    melhor_playlist: list[Faixa] = []
    melhor_fitness = -math.inf
    for _ in range(N_EXECUCOES):
        playlist = funcao(catalogo, perfil)
        valor = fitness(playlist, perfil)
        valores.append(valor)
        if valor > melhor_fitness:
            melhor_fitness = valor
            melhor_playlist = playlist

    return {
        "algoritmo": nome,
        "media": statistics.mean(valores),
        "desvio": statistics.pstdev(valores),
        "min": min(valores),
        "max": max(valores),
        "execucoes": N_EXECUCOES,
        "melhor_playlist": [f.nome for f in melhor_playlist],
        "melhor_fitness": melhor_fitness,
    }


def executar() -> None:
    """Fluxo completo: roda os experimentos e grava os artefatos."""
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    random.seed(SEMENTE)

    t_inicio = time.perf_counter()
    catalogo = carregar_catalogo()

    # --- Experimento 4 (baselines, rapidos) e 1 (GA padrao) ----------------
    # Os baselines rodam primeiro porque sao baratos; o resultado do GA com
    # os parametros padrao alimenta ao mesmo tempo o experimento de
    # convergencia e a comparacao final, evitando rodar o mesmo GA duas vezes.
    print("Rodando os baselines...")
    baselines: dict[str, dict] = {}
    for nome_perfil, perfil in PERFIS_EXPERIMENTO.items():
        baselines[nome_perfil] = {
            "aleatoria": _rodar_baseline(
                "aleatoria", playlist_aleatoria, catalogo, perfil),
            "gulosa": _rodar_baseline(
                "gulosa", playlist_gulosa, catalogo, perfil),
        }

    print("Rodando o algoritmo genetico (parametros padrao)...")
    resumo_ga: dict[str, dict] = {}
    for nome_perfil, perfil in PERFIS_EXPERIMENTO.items():
        print("Perfil:", nome_perfil)
        resumo_ga[nome_perfil] = _rodar_ga("genetico", catalogo, perfil)

    # --- Experimento 2: varredura do tamanho da populacao ------------------
    print("Varredura do tamanho da populacao...")
    varredura_populacao: list[dict] = []
    for tamanho in VARREDURA_POPULACAO:
        print("Populacao:", tamanho)
        linha: dict = {"populacao": tamanho, "perfis": {}}
        medias: list[float] = []
        for nome_perfil, perfil in PERFIS_EXPERIMENTO.items():
            r = _rodar_ga("genetico", catalogo, perfil, tam_populacao=tamanho)
            linha["perfis"][nome_perfil] = r["media"]
            medias.append(r["media"])
        linha["media_geral"] = statistics.mean(medias)
        varredura_populacao.append(linha)

    # --- Experimento 3: varredura da taxa de mutacao ------------------------
    print("Varredura da taxa de mutacao...")
    varredura_mutacao: list[dict] = []
    for taxa in VARREDURA_MUTACAO:
        print("Taxa de mutacao:", taxa)
        linha = {"taxa_mutacao": taxa, "perfis": {}}
        medias = []
        for nome_perfil, perfil in PERFIS_EXPERIMENTO.items():
            r = _rodar_ga("genetico", catalogo, perfil, taxa_mutacao=taxa)
            linha["perfis"][nome_perfil] = r["media"]
            medias.append(r["media"])
        linha["media_geral"] = statistics.mean(medias)
        varredura_mutacao.append(linha)

    # --- Consolida resultados -----------------------------------------------
    resultados = {
        "parametros": {
            "pesos": {
                "compatibilidade": PESO_COMPATIBILIDADE,
                "duracao": PESO_DURACAO,
                "progressao": PESO_PROGRESSAO,
                "transicoes": PESO_TRANSICOES,
                "diversidade": PESO_DIVERSIDADE,
            },
            "algoritmo_genetico": {
                "populacao": POPULACAO,
                "taxa_crossover": TAXA_CROSSOVER,
                "taxa_mutacao": TAXA_MUTACAO,
                "elite": ELITE,
                "geracoes": N_GERACOES,
                "estagnacao": ESTAGNACAO,
                "torneio_k": TORNEIO_K,
            },
            "n_execucoes": N_EXECUCOES,
            "semente": SEMENTE,
        },
        "perfis": {
            nome: {
                "duracao_alvo_min": perfil.duracao_alvo_s / 60.0,
                "energia": perfil.energia,
                "valencia": perfil.valencia,
                "acustica": perfil.acustica,
                "dancabilidade": perfil.dancabilidade,
                "andamento": perfil.andamento,
                "progressao": perfil.progressao,
                "n_faixas": n_faixas_por_perfil(catalogo, perfil),
            }
            for nome, perfil in PERFIS_EXPERIMENTO.items()
        },
        "resultados": {
            nome: {
                **{chave: {k: v for k, v in r.items()
                           if k not in ("historico_melhor", "historico_media")}
                   for chave, r in baselines[nome].items()},
                **{"genetico": {k: v for k, v in resumo_ga[nome].items()
                                if k not in ("historico_melhor", "historico_media")}},
            }
            for nome in PERFIS_EXPERIMENTO
        },
        "varreduras": {
            "populacao": varredura_populacao,
            "mutacao": varredura_mutacao,
        },
    }

    caminho_json = os.path.join(DIR_RESULTADOS, "proj4_resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    # Tabela geral: media, desvio, minimo e maximo por perfil e algoritmo.
    caminho_csv = os.path.join(DIR_RESULTADOS, "proj4_tabela.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["perfil", "algoritmo", "media", "desvio", "min", "max"])
        for nome, grupo in resultados["resultados"].items():
            for algoritmo, r in grupo.items():
                escritor.writerow([nome, algoritmo, f"{r['media']:.4f}",
                                   f"{r['desvio']:.4f}", f"{r['min']:.4f}",
                                   f"{r['max']:.4f}"])

    # Tabelas das varreduras.
    nomes_perfis = list(PERFIS_EXPERIMENTO)
    caminho_csv_pop = os.path.join(DIR_RESULTADOS, "proj4_sweep_populacao.csv")
    with open(caminho_csv_pop, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["populacao", *nomes_perfis, "media_geral"])
        for linha in varredura_populacao:
            escritor.writerow([linha["populacao"],
                               *[f"{linha['perfis'][n]:.4f}" for n in nomes_perfis],
                               f"{linha['media_geral']:.4f}"])

    caminho_csv_mut = os.path.join(DIR_RESULTADOS, "proj4_sweep_mutacao.csv")
    with open(caminho_csv_mut, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["taxa_mutacao", *nomes_perfis, "media_geral"])
        for linha in varredura_mutacao:
            escritor.writerow([f"{linha['taxa_mutacao']:.2f}",
                               *[f"{linha['perfis'][n]:.4f}" for n in nomes_perfis],
                               f"{linha['media_geral']:.4f}"])

    # --- Figuras ------------------------------------------------------------
    plotar_convergencia(
        resumo_ga, os.path.join(DIR_FIGURAS, "proj4_fig1_convergencia.png"))
    plotar_varredura(
        [str(t) for t in VARREDURA_POPULACAO],
        {n: [l["perfis"][n] for l in varredura_populacao] for n in nomes_perfis},
        os.path.join(DIR_FIGURAS, "proj4_fig2_populacao.png"),
        "Fitness medio por tamanho de populacao", "Tamanho da populacao")
    plotar_varredura(
        [f"{int(t * 100)}%" for t in VARREDURA_MUTACAO],
        {n: [l["perfis"][n] for l in varredura_mutacao] for n in nomes_perfis},
        os.path.join(DIR_FIGURAS, "proj4_fig3_mutacao.png"),
        "Fitness medio por taxa de mutacao", "Taxa de mutacao")
    plotar_comparacao(
        {"Aleatoria": [baselines[n]["aleatoria"]["media"] for n in nomes_perfis],
         "Gulosa": [baselines[n]["gulosa"]["media"] for n in nomes_perfis],
         "Genetico": [resumo_ga[n]["media"] for n in nomes_perfis]},
        nomes_perfis,
        os.path.join(DIR_FIGURAS, "proj4_fig4_comparacao.png"))

    # --- Playlists vencedoras ------------------------------------------------
    por_nome = {f.nome: f for f in catalogo}
    for nome, perfil in PERFIS_EXPERIMENTO.items():
        r = resumo_ga[nome]
        playlist = [por_nome[n] for n in r["melhor_playlist"] if n in por_nome]
        print(f"\nPerfil: {nome}")
        imprimir_playlist(playlist, r["melhor_fitness"], perfil)

    t_fim = time.perf_counter()
    print("Artefatos gravados em results/ e figures/.")
    print(f"Tempo total: {t_fim - t_inicio:.1f}s")


# ---------------------------------------------------------------------------
# 10. FIGURAS
# ---------------------------------------------------------------------------


def plotar_convergencia(resumo_ga: dict[str, dict], caminho: str) -> None:
    """Desenha o melhor e a media de fitness por geracao, um painel por perfil.

    A figura mostra a populacao evoluindo: o melhor individuo sobe em degraus
    a cada melhoria aceita e a media acompanha mais lentamente, ilustrando a
    pressao seletiva do torneio e o efeito do elitismo.
    """
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for eixo, (nome, r) in zip(eixos, resumo_ga.items()):
        eixo.plot(r["historico_melhor"], label="melhor", linewidth=1.5)
        eixo.plot(r["historico_media"], label="media",
                  linewidth=1.5, alpha=0.7)
        eixo.set_title(nome)
        eixo.set_xlabel("Geracao")
        eixo.set_ylabel("Fitness")
        eixo.legend()
        eixo.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plotar_varredura(rotulos: list[str], series: dict[str, list[float]],
                     caminho: str, titulo: str, rotulo_x: str) -> None:
    """Desenha barras agrupadas por perfil para uma varredura de parametro."""
    x = list(range(len(rotulos)))
    n_series = len(series)
    largura = 0.75 / n_series

    plt.figure(figsize=(8, 5))
    for i, (nome, valores) in enumerate(series.items()):
        deslocamento = (i - (n_series - 1) / 2) * largura
        plt.bar([xi + deslocamento for xi in x], valores, largura, label=nome)
    plt.xticks(x, rotulos)
    plt.xlabel(rotulo_x)
    plt.ylabel("Fitness medio")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def plotar_comparacao(series: dict[str, list[float]], nomes_perfis: list[str],
                      caminho: str) -> None:
    """Desenha a comparacao entre baselines e algoritmo genetico.

    As barras sao agrupadas por perfil, o que mostra em que situacoes a
    evolucao mais se distancia dos baselines.
    """
    x = list(range(len(nomes_perfis)))
    n_series = len(series)
    largura = 0.75 / n_series

    plt.figure(figsize=(8, 5))
    for i, (nome, valores) in enumerate(series.items()):
        deslocamento = (i - (n_series - 1) / 2) * largura
        plt.bar([xi + deslocamento for xi in x], valores, largura, label=nome)
    plt.xticks(x, nomes_perfis)
    plt.xlabel("Perfil")
    plt.ylabel("Fitness medio")
    plt.title("Comparacao entre baselines e algoritmo genetico")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


if __name__ == "__main__":
    executar()
