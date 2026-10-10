#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 3 - BUSCA EM AMBIENTES COMPLEXOS
=========================================
Geracao de playlists com Hill Climbing e Simulated Annealing sobre um
catalogo de cem faixas reais do Spotify.

O problema pode ser resumido assim. Existe uma biblioteca de cem faixas com
atributos numericos (energia, valencia, acustica, dancabilidade e andamento,
alem da duracao) e o usuario descreve o que quer ouvir: a duracao aproximada
da playlist, os valores medios desejados de cada atributo e um padrao de
progressao de energia (constante, crescente, decrescente ou pico no meio).
O programa deve escolher um subconjunto das faixas e a ordem em que elas
aparecem, de modo que a playlist final maximize uma funcao de qualidade que
pondera compatibilidade com o perfil, adequacao da duracao, progressao da
energia, suavidade das transicoes e diversidade de artistas.

O que torna o problema um caso de busca em ambientes complexos e o tamanho
do espaco de estados e a presenca de otimos locais. Uma playlist e uma
permutacao de um subconjunto das cem faixas, entao o numero de candidatos e
astronomico, e qualquer playlist razoavel tem vizinhas muito parecidas com
ela, o que cria muitos picos locais na funcao de qualidade. Por isso o
projeto usa dois algoritmos classicos de busca local vistos em aula: Hill
Climbing e Simulated Annealing. O primeiro sobe ate o pico mais proximo e o
segundo aceita pioras temporarias enquanto a temperatura decai, conseguindo
escapar de otimos locais.

Os dados vieram de fontes reais. As cem faixas foram obtidas no endpoint de
busca da API do Spotify (modo client credentials, mercado brasileiro,
consultas por genero e ano) e as caracteristicas numericas de cada faixa
vieram da API FreqBlog, um servico que reproduz o formato do antigo endpoint
de audio features do Spotify, que esta bloqueado para aplicativos novos desde
2024. O resultado dessa coleta esta no arquivo data/spotify_tracks.json, que
este programa apenas le; nao ha nenhuma chamada de rede nem chave de API
neste codigo. Uma copia do arquivo esta disponivel na nuvem no endereco
configurado em URL_DATASET_NUVEM, logo abaixo.

DEPENDENCIAS
  matplotlib (figuras). Nada alem da biblioteca padrao.

EXECUCAO
  python proj3_busca_complexa.py
  O programa pede o perfil musical no proprio terminal.
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

# Caminho do dataset. O arquivo e gerado uma unica vez, a partir da API real
# do Spotify e da API FreqBlog, e depois passa a ser apenas lido. Se o arquivo
# nao existir, o programa avisa e encerra, sem tentar acessar a internet.
ARQUIVO_DATASET = os.path.join(DIR_DADOS, "spotify_tracks.json")

# Link publico do dataset na nuvem, para o professor baixar o arquivo caso a
# copia local nao acompanhe a entrega.
URL_DATASET_NUVEM = ""

# Semente para tornar o experimento reproduzivel.
SEMENTE = 42

# Quantas vezes cada algoritmo e executado no experimento final.
N_EXECUCOES = 5

# Pesos de cada componente da funcao de qualidade. A soma e exatamente 1,0.
# Mudancas nesses pesos alteram o comportamento da busca e rendem um bom
# experimento para a secao de resultados do relatorio.
PESO_COMPATIBILIDADE = 0.30
PESO_DURACAO = 0.20
PESO_PROGRESSAO = 0.25
PESO_TRANSICOES = 0.15
PESO_DIVERSIDADE = 0.10

# Parametros do Simulated Annealing, calibrados na varredura documentada em
# results/proj3_ajuste_sa.md para deixar o algoritmo em patamar proximo ao do
# Hill Climbing. Com os valores da aula 08 (50.0, 0.95 e 0.01) a temperatura
# alta aceitava quase toda piora nas primeiras iteracoes e o resultado ficava
# proximo do baseline aleatorio; o resfriamento lento corrige isso.
TEMPERATURA_INICIAL = 0.5
TAXA_RESFRIAMENTO = 0.99
TEMPERATURA_MINIMA = 0.001

# Quantos vizinhos sao gerados em cada passo da busca local. Valores maiores
# deixam a busca mais cara porem mais informada.
N_VIZINHOS = 20

# Tamanho maximo de uma playlist, medido em numero de faixas. O algoritmo
# respeita a duracao alvo cortando faixas, mas este teto evita playlists
# gigantes quando o usuario pede algo muito longo.
TETO_FAIXAS = 30

# Padroes de progressao de energia aceitos na interface.
PROGRESSOES = ("constante", "crescente", "decrescente", "pico")


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
    atributo. A duracao alvo nunca e None porque uma playlist precisa de um
    tamanho aproximado para o corte de faixas fazer sentido.
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


# ---------------------------------------------------------------------------
# 3. INTERFACE DE TERMINAL
# ---------------------------------------------------------------------------


def _ler_float(mensagem: str, minimo: float, maximo: float,
               permite_pular: bool = False) -> float | None:
    """Le um numero do terminal com validacao em laço, ate o valor ser valido.

    Quando permite_pular e verdadeiro, uma linha em branco devolve None e o
    usuario fica livre para ignorar aquele atributo do perfil.
    """
    while True:
        entrada = input(mensagem).strip().replace(",", ".")
        if entrada == "" and permite_pular:
            return None
        try:
            valor = float(entrada)
        except ValueError:
            print("Valor invalido, tente de novo.")
            continue
        if not minimo <= valor <= maximo:
            print(f"Valor fora do intervalo [{minimo}, {maximo}], tente de novo.")
            continue
        return valor


def ler_perfil_usuario(catalogo: list[Faixa]) -> Perfil:
    """Pergunta ao usuario o perfil musical desejado e devolve um Perfil.

    A duracao alvo e obrigatoria e o restante pode ser pulado com Enter.
    Os limites de andamento cobrem praticamente todo o catalogo, que fica
    entre oitenta e cento e setenta batidas por minuto.
    """
    print()
    print("=" * 46)
    print("       GERADOR DE PLAYLISTS - BUSCA COMPLEXA")
    print("=" * 46)
    print()
    print("Descreva a playlist desejada. Nos campos de atributo, um valor em")
    print("branco significa que voce nao se importa com aquela caracteristica.")

    duracao_min = _ler_float("Duracao alvo em minutos: ", 5.0, 180.0)
    assert duracao_min is not None

    energia = _ler_float("Energia desejada (0.0 a 1.0, Enter para ignorar): ",
                         0.0, 1.0, permite_pular=True)
    valencia = _ler_float("Positividade desejada (0.0 a 1.0, Enter para ignorar): ",
                          0.0, 1.0, permite_pular=True)
    acustica = _ler_float("Acustica desejada (0.0 a 1.0, Enter para ignorar): ",
                          0.0, 1.0, permite_pular=True)
    danca = _ler_float("Dancabilidade desejada (0.0 a 1.0, Enter para ignorar): ",
                       0.0, 1.0, permite_pular=True)
    andamento = _ler_float("Andamento desejado em BPM (Enter para ignorar): ",
                           60.0, 220.0, permite_pular=True)

    print()
    print("Padrao de progressao da energia:")
    for i, nome in enumerate(PROGRESSOES, start=1):
        print(f"  ({i}) {nome}")
    escolha = _ler_float("Escolha (1 a 4): ", 1.0, 4.0)
    assert escolha is not None

    return Perfil(
        duracao_alvo_s=duracao_min * 60.0,
        energia=energia,
        valencia=valencia,
        acustica=acustica,
        dancabilidade=danca,
        andamento=andamento,
        progressao=PROGRESSOES[int(escolha) - 1],
    )


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
# 4. FUNCAO DE FITNESS
# ---------------------------------------------------------------------------
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
    return max(0.0, min(1.0, nota)) # garante que a nota fique entre 0 e 1

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
    """Soma ponderada das cinco notas, o numero que a busca maximiza.

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
# 5. VIZINHANCA
# ---------------------------------------------------------------------------


def vizinhanca(playlist: list[Faixa], catalogo: list[Faixa],
               k: int = N_VIZINHOS) -> list[list[Faixa]]:
    """Gera k vizinhos da playlist atual com tres tipos de movimento.

    A troca inverte duas posicoes da playlist, a substituicao troca uma
    faixa por outra do catalogo que ainda nao esteja na playlist e a inversao
    inverte todo o trecho entre dois cortes. Cada vizinho e uma copia nova da
    lista, nunca a mesma referencia. Os mesmos movimentos servirao de mutacao
    no algoritmo genetico do projeto 4.
    """
    vizinhos = []
    disponiveis = set(catalogo) - set(playlist)

    for _ in range(k):
        movimento = random.choice(["swap", "replace", "invert"])
        if len(playlist) < 2:
            movimento = "replace"

        if movimento == "swap":
            posicoes = random.sample(range(len(playlist)), 2)
            vizinho = list(playlist)
            vizinho[posicoes[0]], vizinho[posicoes[1]] = (
                vizinho[posicoes[1]], vizinho[posicoes[0]])
        elif movimento == "replace":
            vizinho = list(playlist)
            if disponiveis:
                posicao = random.randrange(len(playlist))
                vizinho[posicao] = random.choice(list(disponiveis))
        else:
            cortes = sorted(random.sample(range(len(playlist) + 1), 2))
            vizinho = list(playlist)
            vizinho[cortes[0]:cortes[1]] = reversed(vizinho[cortes[0]:cortes[1]])

        vizinhos.append(vizinho)

    return vizinhos


# ---------------------------------------------------------------------------
# 6. ALGORITMOS DE BUSCA EM AMBIENTES COMPLEXOS
# ---------------------------------------------------------------------------
# As funcoes de busca devolvem uma tupla com a melhor playlist encontrada, o
# fitness dela e o historico de fitness a cada passo, usado nas figuras.


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
    progressao pedido, o que deixa a comparacao com a busca local mais justa.
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


def hill_climbing(playlist_inicial: list[Faixa], catalogo: list[Faixa],
                  perfil: Perfil) -> tuple[list[Faixa], float, list[float]]:
    """Busca local classica que sobe enquanto existir vizinho melhor.

    Segue o esqueleto do exemplo da aula 08, com a vizinhanca de playlists no
    lugar de numeros vizinhos e o fitness como funcao objetivo. A cada passo
    avalia todos os vizinhos e move para o melhor apenas se ele superar o
    atual, guardando o historico de fitness para a figura. Quando nenhum
    vizinho melhora, a busca parou em um otimo local. Para contornar os
    otimos locais, o experimento roda a funcao varias vezes a partir de
    playlists aleatorias e fica com o melhor resultado, em um esquema de
    reinicio aleatorio.
    """
    atual = list(playlist_inicial)
    fitness_atual = fitness(atual, perfil)
    historico = [fitness_atual]

    while True:
        melhor_vizinho = None
        melhor_fitness = fitness_atual

        for vizinho in vizinhanca(atual, catalogo):
            valor = fitness(vizinho, perfil)
            if valor > melhor_fitness:
                melhor_vizinho = vizinho
                melhor_fitness = valor

        if melhor_vizinho is None:
            break

        atual = melhor_vizinho
        fitness_atual = melhor_fitness
        historico.append(fitness_atual)

    return (atual, fitness_atual, historico)


def simulated_annealing(playlist_inicial: list[Faixa], catalogo: list[Faixa],
                        perfil: Perfil) -> tuple[list[Faixa], float, list[float]]:
    """Busca local com temperatura que aceita pioras temporarias.

    Segue o exemplo da aula 08. A cada passo sorteia um vizinho e calcula o
    delta do fitness; melhoras sao sempre aceitas e pioras entram com
    probabilidade exp(delta / temperatura), que diminui conforme a
    temperatura cai por multiplicacao pela taxa de resfriamento, ate cruzar a
    temperatura minima. O historico guarda o melhor fitness visto ate o
    momento, porque o estado corrente pode estar temporariamente pior.
    """
    atual = list(playlist_inicial)
    fitness_atual = fitness(atual, perfil)
    temperatura = TEMPERATURA_INICIAL
    historico = [fitness_atual]
    melhor_playlist = list(atual)
    melhor_fitness = fitness_atual

    while temperatura > TEMPERATURA_MINIMA:
        vizinhos = vizinhanca(atual, catalogo)
        if not vizinhos:
            break
        proximo = random.choice(vizinhos)
        proximo_fitness = fitness(proximo, perfil)
        delta = proximo_fitness - fitness_atual

        if delta > 0 or random.random() < math.exp(delta / temperatura):
            atual = proximo
            fitness_atual = proximo_fitness

            if fitness_atual > melhor_fitness:
                melhor_playlist = list(atual)
                melhor_fitness = fitness_atual

        historico.append(melhor_fitness)
        temperatura *= TAXA_RESFRIAMENTO

    return (melhor_playlist, melhor_fitness, historico)


# ---------------------------------------------------------------------------
# 7. EXPERIMENTO
# ---------------------------------------------------------------------------


def _rodar_algoritmo(nome: str, funcao, catalogo: list[Faixa],
                     perfil: Perfil) -> dict:
    """Roda um algoritmo varias vezes e resume media, desvio e melhor caso."""
    valores: list[float] = []
    melhor_playlist: list[Faixa] = []
    melhor_fitness = -math.inf
    historico: list[float] = []
    for _ in range(N_EXECUCOES):
        inicio = playlist_aleatoria(catalogo, perfil)
        t0 = time.perf_counter()
        playlist, valor, hist = funcao(inicio, catalogo, perfil)
        t1 = time.perf_counter()
        valores.append(valor)
        if valor > melhor_fitness:
            melhor_fitness = valor
            melhor_playlist = playlist
            historico = hist
        print(f"  [{nome}] fitness={valor:.3f}  ({t1 - t0:.2f}s)")

    return {
        "algoritmo": nome,
        "media": statistics.mean(valores),
        "desvio": statistics.pstdev(valores),
        "min": min(valores),
        "max": max(valores),
        "execucoes": N_EXECUCOES,
        "melhor_playlist": [f.nome for f in melhor_playlist],
        "historico": historico,
    }


def executar() -> None:
    """Fluxo completo: le o perfil, roda as buscas e grava os artefatos."""
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    random.seed(SEMENTE)

    catalogo = carregar_catalogo()
    perfil = ler_perfil_usuario(catalogo)

    print()
    print("Rodando as buscas...")
    resumo: dict[str, dict] = {}
    resumo["aleatoria"] = _rodar_algoritmo(
        "aleatoria", lambda ini, cat, per: (ini, fitness(ini, per), []),
        catalogo, perfil)
    resumo["gulosa"] = _rodar_algoritmo(
        "gulosa", lambda ini, cat, per: (pl := playlist_gulosa(cat, per),
                                         fitness(pl, per), []),
        catalogo, perfil)
    resumo["hill_climbing"] = _rodar_algoritmo(
        "hill_climbing", hill_climbing, catalogo, perfil)
    resumo["simulated_annealing"] = _rodar_algoritmo(
        "simulated_annealing", simulated_annealing, catalogo, perfil)

    melhor_nome = max(resumo, key=lambda n: resumo[n]["max"])
    melhor = resumo[melhor_nome]

    resultados = {
        "perfil": {
            "duracao_alvo_min": perfil.duracao_alvo_s / 60.0,
            "energia": perfil.energia,
            "valencia": perfil.valencia,
            "acustica": perfil.acustica,
            "dancabilidade": perfil.dancabilidade,
            "andamento": perfil.andamento,
            "progressao": perfil.progressao,
        },
        "parametros": {
            "pesos": {
                "compatibilidade": PESO_COMPATIBILIDADE,
                "duracao": PESO_DURACAO,
                "progressao": PESO_PROGRESSAO,
                "transicoes": PESO_TRANSICOES,
                "diversidade": PESO_DIVERSIDADE,
            },
            "simulated_annealing": {
                "temperatura_inicial": TEMPERATURA_INICIAL,
                "taxa_resfriamento": TAXA_RESFRIAMENTO,
                "temperatura_minima": TEMPERATURA_MINIMA,
            },
            "n_vizinhos": N_VIZINHOS,
            "n_execucoes": N_EXECUCOES,
        },
        "resultados": {nome: {k: v for k, v in r.items() if k != "historico"}
                       for nome, r in resumo.items()},
    }

    caminho_json = os.path.join(DIR_RESULTADOS, "proj3_resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    caminho_csv = os.path.join(DIR_RESULTADOS, "proj3_tabela.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["algoritmo", "media", "desvio", "min", "max"])
        for nome, r in resumo.items():
            escritor.writerow([nome, f"{r['media']:.4f}", f"{r['desvio']:.4f}",
                               f"{r['min']:.4f}", f"{r['max']:.4f}"])

    plotar_evolucao(resumo, os.path.join(DIR_FIGURAS, "proj3_fig1_evolucao.png"))
    plotar_progressao(melhor["melhor_playlist"], catalogo, perfil,
                      os.path.join(DIR_FIGURAS, "proj3_fig2_progressao.png"))

    por_nome = {f.nome: f for f in catalogo}
    playlist_final = [por_nome[n] for n in melhor["melhor_playlist"] if n in por_nome]
    imprimir_playlist(playlist_final, melhor["max"], perfil)

    print("Artefatos gravados em results/ e figures/.")
    print("Melhor algoritmo: " + melhor_nome)


# ---------------------------------------------------------------------------
# 8. FIGURAS
# ---------------------------------------------------------------------------


def plotar_evolucao(resumo: dict[str, dict], caminho: str) -> None:
    """Desenha o fitness ao longo dos passos de cada busca local.

    A figura mostra as curvas de Hill Climbing e Simulated Annealing subindo
    ate estacionar, o que ilustra a convergencia dos dois algoritmos.
    """
    plt.figure(figsize=(8, 5))
    for nome, r in resumo.items():
        hist = r.get("historico") or []
        if hist:
            plt.plot(hist, label=nome, linewidth=1.5)
    plt.xlabel("Passo da busca")
    plt.ylabel("Fitness")
    plt.title("Evolucao do fitness durante a busca")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def plotar_progressao(nomes: list[str], catalogo: list[Faixa], perfil: Perfil,
                      caminho: str) -> None:
    """Desenha a curva de energia da melhor playlist encontrada."""
    por_nome = {f.nome: f for f in catalogo}
    playlist = [por_nome[n] for n in nomes if n in por_nome]
    if not playlist:
        return
    plt.figure(figsize=(8, 5))
    plt.plot([f.energia for f in playlist], marker="o", linewidth=1.5)
    plt.xlabel("Posicao na playlist")
    plt.ylabel("Energia")
    plt.title(f"Progressao de energia ({perfil.progressao})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


if __name__ == "__main__":
    executar()
