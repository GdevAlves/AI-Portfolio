#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 6 - BANCO DE CONHECIMENTO
=================================
Assessor de playlists baseado em regras sobre o mesmo catalogo de cem faixas
reais do Spotify usado nos projetos 3 e 4. Onde a busca local e o algoritmo
genetico tratam a playlist como um problema de otimizacao numerica, este
projeto trata a recomendacao como um problema de INFERENCIA SIMBOLICA: uma
base de conhecimento com fatos sobre as faixas e sobre o usuario, regras de
producao no formato de clausulas de Horn, um motor de encadeamento
progressivo (forward chaining) para derivar as exigencias musicais e um
motor de encadeamento regressivo (backward chaining) que explica cada
recomendacao com uma arvore de prova.

PROBLEMA FORMAL (Russell & Norvig, AIMA 4a ed., capitulo 7 - Agentes Logicos)
  - Agente      : assessor de playlists baseado em conhecimento
  - Percepcao   : fatos do usuario (ocasiao, humor, periodo e duracao alvo)
  - Base de     : fatos simbolicos sobre as 100 faixas (energia alta/baixa,
    conhecimento  andamento rapido/lento, positividade, acustica, danca,
                  duracao curta/longa) e regras Horn que ligam o contexto do
                  usuario a exigencias musicais e faixas que as atendem
  - Inferencia  : encadeamento progressivo (deriva exigencias e recomendacoes)
                  e regressivo (prova e explica cada recomendacao)
  - Acoes       : selecionar faixas e ordena-las pelo padrao de progressao
  - Objetivo    : playlist que atenda as exigencias derivadas e a duracao alvo
  - Restricao   : "dura" para a representacao (uma faixa nao aparece duas
                  vezes) e "mole" para a duracao, perseguida pelo corte de
                  faixas durante a selecao

POR QUE UM BANCO DE CONHECIMENTO?
  O problema pede explicacao. Um filtro numerico devolve uma lista de faixas;
  um sistema baseado em conhecimento devolve a lista E a cadeia de regras que
  justifica cada faixa ("esta faixa entrou porque academia exige energia alta
  e ela tem energia 0.81"). Essa capacidade de explicar o proprio raciocinio
  e a contribuicao central do paradigma simbolico e o que diferencia este
  projeto dos anteriores no portfolio.

REPRESENTACAO
  Cada fato e uma string do tipo "predicado(argumento)", como em logica de
  primeira ordem aterrada: "tem(energia_alta, id_da_faixa)" descreve o
  catalogo, "ocasiao(academia)" descreve o usuario e "exige(energia_alta)"
  e derivado por inferencia. Cada regra e uma clausula de Horn com um
  consequente e um conjunto de antecedentes, todos fatos aterrados, com uma
  prioridade que pondera a selecao final. As regras de recomendacao sao
  instanciadas uma vez por faixa e por exgencia, totalizando centenas de
  regras aterradas, exatamente como um KB proposicional derivado de uma
  descricao em primeira ordem.

EXPERIMENTOS
  1. Tres cenarios de usuario (academia, estudo e festa) com metricas de
     inferencia: regras instanciadas, regras disparadas e fatos derivados.
  2. Cobertura: fracao das faixas da playlist que atende cada exgencia.
  3. Distancia media dos atributos da playlist aos alvos numericos de cada
     exgencia, comparada com os baselines aleatorio e guloso.
  4. Arvore de prova impressa para a primeira faixa de cada cenario.

DEPENDENCIAS
  matplotlib (figuras). Nada alem da biblioteca padrao.

EXECUCAO
  python proj6_banco_conhecimento.py
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

# Caminho do dataset, o mesmo dos projetos 3 e 4.
ARQUIVO_DATASET = os.path.join(DIR_DADOS, "spotify_tracks.json")
URL_DATASET_NUVEM = ""

# Semente para tornar o experimento reproduzivel.
SEMENTE = 42

# Quantas vezes cada baseline e executado no experimento.
N_EXECUCOES = 5

# Tamanho maximo de uma playlist, em numero de faixas.
TETO_FAIXAS = 30

# Limiares que transformam os atributos numericos do catalogo em predicados
# simbolicos. Os valores seguem convencoes de literatura sobre audio features
# (energia e positividade divididas em tercos, andamento dividido em torno de
# 95/120 BPM, acustica e dancabilidade pela metade da escala).
LIMIARES = {
    "energia_alta": 0.66,
    "energia_baixa": 0.33,
    "valencia_alta": 0.60,
    "valencia_baixa": 0.40,
    "acustica": 0.50,
    "dancavel": 0.65,
    "andamento_rapido": 120.0,
    "andamento_lento": 95.0,
    "duracao_curta": 180.0,
    "duracao_longa": 240.0,
}

# Traducao de cada exgencia simbolica em um alvo numerico por atributo,
# usada na metrica de distancia que compara regras com baselines numericos.
ALVOS_NUMERICOS: dict[str, tuple[str, float]] = {
    "energia_alta": ("energia", 0.80),
    "energia_baixa": ("energia", 0.30),
    "valencia_alta": ("valencia", 0.70),
    "valencia_baixa": ("valencia", 0.30),
    "acustica": ("acustica", 0.60),
    "dancavel": ("dancabilidade", 0.70),
    "andamento_rapido": ("andamento", 130.0),
    "andamento_lento": ("andamento", 85.0),
}

# Padroes de progressao aceitos.
PROGRESSOES = ("constante", "crescente", "decrescente", "pico")


@dataclass(frozen=True)
class Faixa:
    """Uma faixa do catalogo com os atributos usados pelas regras."""

    id: str
    nome: str
    artistas: tuple[str, ...]
    album: str
    ano: int
    duracao_s: float
    energia: float
    valencia: float
    acustica: float
    dancabilidade: float
    andamento: float


@dataclass(frozen=True)
class Regra:
    """Clausula de Horn: se todos os antecedentes, entao o consequente.

    A prioridade pondera a selecao final: exigencias vindas de regras de
    prioridade alta valem mais na pontuacao das faixas.
    """

    nome: str
    antecedentes: frozenset[str]
    consequente: str
    prioridade: int = 1


@dataclass(frozen=True)
class Cenario:
    """Cenario de usuario: fatos iniciais, duracao alvo e progressao."""

    nome: str
    fatos: frozenset[str]
    duracao_alvo_s: float
    progressao: str


# ---------------------------------------------------------------------------
# 2. CARREGAMENTO DO DATASET
# ---------------------------------------------------------------------------


def carregar_catalogo(caminho: str = ARQUIVO_DATASET) -> list[Faixa]:
    """Le o arquivo JSON com as cem faixas e devolve uma lista de Faixa."""
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
            energia=item["energy"],
            valencia=item["valence"],
            acustica=item["acousticness"],
            dancabilidade=item["danceability"],
            andamento=item["tempo"],
        ))

    return catalogo


def _total_playlist(playlist: list[Faixa]) -> float:
    """Duracao total em segundos de uma lista de faixas."""
    return sum(f.duracao_s for f in playlist)


def formatar_tempo(segundos: float) -> str:
    """Converte segundos em texto mm:ss."""
    minutos = int(segundos // 60)
    restante = int(round(segundos % 60))
    if restante == 60:
        minutos += 1
        restante = 0
    return f"{minutos:02d}:{restante:02d}"


# ---------------------------------------------------------------------------
# 3. BASE DE FATOS
# ---------------------------------------------------------------------------


def fatos_faixa(faixa: Faixa) -> set[str]:
    """Traduz os atributos numericos de uma faixa em predicados simbolicos.

    Cada limiar da configuracao vira uma checagem de corte; o fato resultante
    tem a forma "tem(predicado, id_da_faixa)" e entra na base de conhecimento
    como fato aterrado. Energia e andamento produzem pares complementares
    (alta/baixa, rapido/lento); faixas intermediarias simplesmente nao tem o
    predicado dos extremos, o que a logica classica trata como falso.
    """
    fatos: set[str] = set()
    regras_de_corte = {
        "energia_alta": faixa.energia >= LIMIARES["energia_alta"],
        "energia_baixa": faixa.energia <= LIMIARES["energia_baixa"],
        "valencia_alta": faixa.valencia >= LIMIARES["valencia_alta"],
        "valencia_baixa": faixa.valencia <= LIMIARES["valencia_baixa"],
        "acustica": faixa.acustica >= LIMIARES["acustica"],
        "dancavel": faixa.dancabilidade >= LIMIARES["dancavel"],
        "andamento_rapido": faixa.andamento >= LIMIARES["andamento_rapido"],
        "andamento_lento": faixa.andamento <= LIMIARES["andamento_lento"],
        "duracao_curta": faixa.duracao_s <= LIMIARES["duracao_curta"],
        "duracao_longa": faixa.duracao_s >= LIMIARES["duracao_longa"],
    }
    for predicado, verdadeiro in regras_de_corte.items():
        if verdadeiro:
            fatos.add(f"tem({predicado}, {faixa.id})")
    return fatos


# Cenarios do experimento: academia pede energia e ritmo, estudo pede
# acustica, festa pede danca e positividade. Os fatos sao os mesmos que um
# usuario daria numa conversa com o assessor.
CENARIOS: dict[str, Cenario] = {
    "Academia, 60 min, crescente": Cenario(
        nome="academia",
        fatos=frozenset({"ocasiao(academia)", "humor(energico)",
                         "periodo(manha)", "duracao_alvo(60)"}),
        duracao_alvo_s=60 * 60.0,
        progressao="crescente",
    ),
    "Estudo, 45 min, constante": Cenario(
        nome="estudo",
        fatos=frozenset({"ocasiao(estudo)", "humor(relaxado)",
                         "periodo(noite)", "duracao_alvo(45)"}),
        duracao_alvo_s=45 * 60.0,
        progressao="constante",
    ),
    "Festa, 90 min, pico": Cenario(
        nome="festa",
        fatos=frozenset({"ocasiao(festa)", "humor(alegre)",
                         "periodo(noite)", "duracao_alvo(90)"}),
        duracao_alvo_s=90 * 60.0,
        progressao="pico",
    ),
}


# ---------------------------------------------------------------------------
# 4. REGRAS
# ---------------------------------------------------------------------------


def regras_preferencia() -> list[Regra]:
    """Regras que ligam o contexto do usuario a exigencias musicais.

    A prioridade codifica o senso comum do dominio: a ocasiao vale mais que o
    humor, que vale mais que o periodo. As exigencias sao os predicados que
    as regras de recomendacao vao casar contra os fatos das faixas.
    """
    return [
        Regra("R1_academia_energia", frozenset({"ocasiao(academia)"}),
              "exige(energia_alta)", 3),
        Regra("R2_academia_andamento", frozenset({"ocasiao(academia)"}),
              "exige(andamento_rapido)", 3),
        Regra("R3_festa_danca", frozenset({"ocasiao(festa)"}),
              "exige(dancavel)", 3),
        Regra("R4_festa_valencia", frozenset({"ocasiao(festa)"}),
              "exige(valencia_alta)", 3),
        Regra("R5_estudo_acustica", frozenset({"ocasiao(estudo)"}),
              "exige(acustica)", 3),
        Regra("R6_alegre_valencia", frozenset({"humor(alegre)"}),
              "exige(valencia_alta)", 2),
        Regra("R7_triste_energia", frozenset({"humor(triste)"}),
              "exige(energia_baixa)", 2),
        Regra("R8_triste_acustica", frozenset({"humor(triste)"}),
              "exige(acustica)", 2),
        Regra("R9_relaxado_acustica", frozenset({"humor(relaxado)"}),
              "exige(acustica)", 2),
        Regra("R10_energico_energia", frozenset({"humor(energico)"}),
              "exige(energia_alta)", 2),
        Regra("R11_manha_andamento", frozenset({"periodo(manha)"}),
              "exige(andamento_rapido)", 1),
        Regra("R12_noite_valencia", frozenset({"periodo(noite)"}),
              "exige(valencia_baixa)", 1),
        Regra("R13_noite_acustica", frozenset({"periodo(noite)"}),
              "exige(acustica)", 1),
    ]


def regras_recomendacao(catalogo: list[Faixa]) -> list[Regra]:
    """Instancia uma regra por faixa e por exgencia (aterramento).

    Para cada par (faixa, predicado) cria a clausula "atende(faixa, P) se
    exige(P) e tem(P, faixa)". Com cem faixas e oito predicados candidatos
    sao oitocentas regras aterradas, cada uma com nome proprio, o que deixa a
    explicacao regressiva localizar exatamente a cadeia que justifica cada
    faixa da playlist.
    """
    predicados = list(ALVOS_NUMERICOS)
    regras: list[Regra] = []
    for faixa in catalogo:
        for predicado in predicados:
            regras.append(Regra(
                nome=f"atende({faixa.nome[:24]}, {predicado})",
                antecedentes=frozenset({
                    f"exige({predicado})", f"tem({predicado}, {faixa.id})"}),
                consequente=f"atende({faixa.id}, {predicado})",
                prioridade=1,
            ))
    return regras


# ---------------------------------------------------------------------------
# 5. MOTOR DE INFERENCIA
# ---------------------------------------------------------------------------


class BaseConhecimento:
    """Base de conhecimento com encadeamento progressivo e regressivo.

    Guarda os fatos iniciais, as regras e o registro das derivacoes. O
    encadeamento progressivo repassa a lista de regras ate atingir o ponto
    fixo, disparando toda regra cujos antecedentes estejam na base e cujo
    consequente ainda nao exista; cada derivacao guarda a regra que a
    produziu, o que sustenta a explicacao regressiva depois.
    """

    def __init__(self, fatos: set[str], regras: list[Regra]) -> None:
        self.fatos_iniciais = frozenset(fatos)
        self.fatos: set[str] = set(fatos)
        self.regras = regras
        self.derivacoes: dict[str, Regra] = {}
        self.disparos: list[Regra] = []

    def encadeamento_progressivo(self) -> None:
        """Deriva todos os fatos possiveis ate o ponto fixo.

        Cada passada varre as regras ainda nao disparadas; enquanto ao menos
        uma derivacao nova aparecer, uma nova passada acontece. Como as
        regras de preferencia antecedem as regras aterradas na lista, as
        exigencias ja estao na base quando as regras de recomendacao sao
        testadas, e o laco termina em poucas passadas.
        """
        while True:
            mudou = False
            for regra in self.regras:
                if regra in self.disparos:
                    continue
                if regra.consequente in self.fatos:
                    continue
                if regra.antecedentes.issubset(self.fatos):
                    self.fatos.add(regra.consequente)
                    self.derivacoes[regra.consequente] = regra
                    self.disparos.append(regra)
                    mudou = True
            if not mudou:
                break

    def explicar(self, objetivo: str, profundidade: int = 0,
                 max_profundidade: int = 6) -> list[str]:
        """Prova regressiva de um fato, devolvendo a arvore de prova em texto.

        Um fato fornecido encerra o galho; um fato derivado e justificado
        pela regra que o produziu, e cada antecedente da regra e provado
        recursivamente. O limite de profundidade protege contra ciclos.
        """
        linhas: list[str] = []
        prefixo = "  " * profundidade
        if objetivo in self.fatos_iniciais:
            linhas.append(f"{prefixo}{objetivo}  [fato fornecido]")
            return linhas
        if profundidade >= max_profundidade:
            linhas.append(f"{prefixo}{objetivo}  [limite de prova]")
            return linhas
        regra = self.derivacoes.get(objetivo)
        if regra is None:
            linhas.append(f"{prefixo}{objetivo}  [indemonstravel]")
            return linhas
        linhas.append(f"{prefixo}{objetivo}  <- regra {regra.nome}")
        for antecedente in sorted(regra.antecedentes):
            linhas.extend(self.explicar(antecedente, profundidade + 1,
                                        max_profundidade))
        return linhas


# ---------------------------------------------------------------------------
# 6. SELECAO E ORDENACAO
# ---------------------------------------------------------------------------


def exigencias_derivadas(bc: BaseConhecimento) -> list[str]:
    """Lista as exigencias musicais derivadas, ordenadas por prioridade."""
    pares: list[tuple[int, str]] = []
    for fato in bc.fatos:
        if fato.startswith("exige("):
            predicado = fato[len("exige("):-1]
            prioridade = bc.derivacoes[fato].prioridade
            pares.append((prioridade, predicado))
    return [p for _, p in sorted(pares, reverse=True)]


def pontuar_faixa(faixa: Faixa, bc: BaseConhecimento) -> float:
    """Pontua a faixa pela soma das prioridades das exigencias que atende."""
    nota = 0.0
    prefixo = f"atende({faixa.id}, "
    for fato in bc.fatos:
        if fato.startswith(prefixo):
            predicado = fato[len(prefixo):-1]
            exigencia = f"exige({predicado})"
            if exigencia in bc.fatos:
                nota += bc.derivacoes[exigencia].prioridade
    return nota


def ordenar_por_progressao(faixas: list[Faixa], padrao: str) -> list[Faixa]:
    """Ordena a playlist pelo padrao de progressao de energia pedido.

    Crescente e decrescente ordenam pela energia; o pico coloca a faixa mais
    energetica no centro e alterna as seguintes para as pontas; o constante
    mantem a ordem de pontuacao, que ja e a mais relevante quando a forma nao
    importa.
    """
    if padrao == "crescente":
        return sorted(faixas, key=lambda f: f.energia)
    if padrao == "decrescente":
        return sorted(faixas, key=lambda f: f.energia, reverse=True)
    if padrao == "pico":
        ordenadas = sorted(faixas, key=lambda f: f.energia, reverse=True)
        resultado: list[Faixa] = [ordenadas[0]] if ordenadas else []
        esquerda, direita = [], []
        for i, faixa in enumerate(ordenadas[1:], start=1):
            (direita if i % 2 == 0 else esquerda).append(faixa)
        return esquerda + resultado + direita
    return list(faixas)


def montar_playlist(catalogo: list[Faixa], bc: BaseConhecimento,
                    cenario: Cenario) -> list[Faixa]:
    """Seleciona e ordena as faixas recomendadas pela base de conhecimento.

    As faixas sao pontuadas pela soma das prioridades das exigencias
    atendidas e entram em ordem decrescente de pontuacao ate que a proxima
    estoure a duracao alvo; a playlist final recebe a ordenacao do padrao de
    progressao do cenario.
    """
    ordenadas = sorted(catalogo, key=lambda f: pontuar_faixa(f, bc),
                       reverse=True)
    selecionadas: list[Faixa] = []
    for faixa in ordenadas:
        if pontuar_faixa(faixa, bc) <= 0:
            break
        if len(selecionadas) >= TETO_FAIXAS:
            break
        if _total_playlist(selecionadas) + faixa.duracao_s > cenario.duracao_alvo_s:
            continue
        selecionadas.append(faixa)
    return ordenar_por_progressao(selecionadas, cenario.progressao)


# ---------------------------------------------------------------------------
# 7. BASELINES
# ---------------------------------------------------------------------------


def playlist_aleatoria(catalogo: list[Faixa], cenario: Cenario) -> list[Faixa]:
    """Baseline aleatorio: embaralha o catalogo e preenche ate a duracao."""
    faixas = list(catalogo)
    random.shuffle(faixas)
    playlist: list[Faixa] = []
    for faixa in faixas:
        if len(playlist) >= TETO_FAIXAS:
            break
        if _total_playlist(playlist) + faixa.duracao_s <= cenario.duracao_alvo_s:
            playlist.append(faixa)
    return playlist


def playlist_gulosa(catalogo: list[Faixa], cenario: Cenario,
                    alvos: dict[str, float]) -> list[Faixa]:
    """Baseline guloso: escolhe sempre a faixa mais proxima dos alvos numericos.

    Usa a mesma traducao exgencia -> alvo numerico da metrica de comparacao,
    o que deixa a disputa com o sistema de regras honesta: o guloso otimiza
    diretamente a metrica numerica, sem nenhum raciocinio simbolico.
    """
    def distancia(faixa: Faixa) -> float:
        dists = []
        for atributo, alvo in alvos.items():
            valor = getattr(faixa, atributo)
            if atributo == "andamento":
                valor /= 100.0
                alvo /= 100.0
            dists.append(abs(valor - alvo))
        return sum(dists) / len(dists) if dists else 0.0

    playlist: list[Faixa] = []
    disponiveis = set(catalogo)
    while disponiveis:
        melhor = min(disponiveis, key=distancia)
        if _total_playlist(playlist) + melhor.duracao_s > cenario.duracao_alvo_s:
            break
        playlist.append(melhor)
        disponiveis.remove(melhor)
    return ordenar_por_progressao(playlist, cenario.progressao)


# ---------------------------------------------------------------------------
# 8. METRICAS
# ---------------------------------------------------------------------------


def alvos_numericos(bc: BaseConhecimento) -> dict[str, float]:
    """Traduz as exigencias derivadas em um dict atributo -> alvo numerico.

    Como as exigencias vem ordenadas por prioridade decrescente, a primeira
    aparicao de cada atributo define o alvo: quando o conhecimento deriva
    exigencias conflitantes sobre o mesmo atributo (positividade alta da
    festa e baixa da noite), a prioridade decide qual alvo vale na metrica.
    """
    alvos: dict[str, float] = {}
    for predicado in exigencias_derivadas(bc):
        atributo, alvo = ALVOS_NUMERICOS[predicado]
        if atributo not in alvos:
            alvos[atributo] = alvo
    return alvos


def distancia_media_alvo(playlist: list[Faixa],
                         alvos: dict[str, float]) -> float:
    """Distancia media dos atributos da playlist aos alvos numericos.

    Para cada faixa, tira a media das distancias absolutas entre os atributos
    pedidos e os alvos (andamento normalizado por 100, como nos projetos
    anteriores); a nota final e a media sobre as faixas. Quanto menor,
    melhor.
    """
    if not playlist or not alvos:
        return 0.0
    distancias = []
    for faixa in playlist:
        da_faixa = []
        for atributo, alvo in alvos.items():
            valor = getattr(faixa, atributo)
            if atributo == "andamento":
                valor /= 100.0
                alvo /= 100.0
            da_faixa.append(abs(valor - alvo))
        distancias.append(sum(da_faixa) / len(da_faixa))
    return sum(distancias) / len(distancias)


def cobertura_playlist(playlist: list[Faixa], bc: BaseConhecimento) -> float:
    """Fracao das exigencias atendidas pela playlist, ponderada por prioridade.

    Para cada exgencia derivada, conta a fracao das faixas da playlist cujo
    fato "atende(faixa, exgencia)" esta na base e tira a media entre as
    exigencias, ponderada pela prioridade da regra que derivou cada uma.
    A ponderacao evita que exigencias de prioridade baixa e mutuamente
    exclusivas (como positividade alta pedida pela festa e positividade
    baixa pedida pela noite) puxem a metrica para baixo na mesma medida que
    as exigencias centrais do cenario. Vale 1 quando todas as faixas atendem
    todas as exigencias.
    """
    exigidas = exigencias_derivadas(bc)
    if not exigidas or not playlist:
        return 0.0
    soma_fracoes = 0.0
    soma_pesos = 0.0
    for predicado in exigidas:
        peso = bc.derivacoes[f"exige({predicado})"].prioridade
        atendidas = sum(
            1 for f in playlist if f"atende({f.id}, {predicado})" in bc.fatos
        )
        soma_fracoes += peso * (atendidas / len(playlist))
        soma_pesos += peso
    return soma_fracoes / soma_pesos


# ---------------------------------------------------------------------------
# 9. EXPERIMENTO
# ---------------------------------------------------------------------------


def _rodar_cenario(catalogo: list[Faixa],
                   cenario: Cenario) -> dict:
    """Roda a inferencia completa de um cenario e mede tudo."""
    t0 = time.perf_counter()

    fatos_iniciais: set[str] = set(cenario.fatos)
    for faixa in catalogo:
        fatos_iniciais |= fatos_faixa(faixa)

    bc = BaseConhecimento(fatos_iniciais,
                          regras_preferencia() + regras_recomendacao(catalogo))
    bc.encadeamento_progressivo()
    t1 = time.perf_counter()

    playlist = montar_playlist(catalogo, bc, cenario)

    alvos = alvos_numericos(bc)
    aleatorias = [playlist_aleatoria(catalogo, cenario)
                  for _ in range(N_EXECUCOES)]
    gulosa = playlist_gulosa(catalogo, cenario, alvos)

    # Cobertura dos baselines usa as mesmas exigencias das regras.
    cov_aleatoria = statistics.mean(cobertura_playlist(p, bc)
                                    for p in aleatorias)
    cov_gulosa = cobertura_playlist(gulosa, bc)

    distancia_regras = distancia_media_alvo(playlist, alvos)
    distancia_aleatoria = statistics.mean(
        distancia_media_alvo(p, alvos) for p in aleatorias)
    distancia_gulosa = distancia_media_alvo(gulosa, alvos)

    # Explicacao da primeira faixa da playlist, para o relatorio.
    explicacao: list[str] = []
    if playlist:
        primeiro = playlist[0]
        fatos_primeira = [f for f in bc.fatos
                          if f.startswith(f"atende({primeiro.id}, ")]
        if fatos_primeira:
            def prioridade_fato(fato: str) -> int:
                predicado = fato.split(", ", 1)[1][:-1]
                return bc.derivacoes[f"exige({predicado})"].prioridade
            explicacao = bc.explicar(max(fatos_primeira,
                                         key=prioridade_fato))

    derivados = len(bc.fatos) - len(bc.fatos_iniciais)
    return {
        "cenario": cenario.nome,
        "regras_instanciadas": len(bc.regras),
        "regras_disparadas": len(bc.disparos),
        "fatos_derivados": derivados,
        "exigencias": exigencias_derivadas(bc),
        "tempo_inferencia_s": t1 - t0,
        "playlist": [f.nome for f in playlist],
        "duracao": formatar_tempo(_total_playlist(playlist)),
        "explicacao": explicacao,
        "metricas": {
            "cobertura_regras": cobertura_playlist(playlist, bc),
            "cobertura_gulosa": cov_gulosa,
            "cobertura_aleatoria": cov_aleatoria,
            "distancia_regras": distancia_regras,
            "distancia_gulosa": distancia_gulosa,
            "distancia_aleatoria": distancia_aleatoria,
        },
    }


def executar() -> None:
    """Fluxo completo: roda os cenarios, grava artefatos e imprime provas."""
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    random.seed(SEMENTE)

    catalogo = carregar_catalogo()
    resumos: dict[str, dict] = {}
    for rotulo, cenario in CENARIOS.items():
        print("Cenario:", rotulo)
        resumos[rotulo] = _rodar_cenario(catalogo, cenario)
        r = resumos[rotulo]
        print(f"  regras instanciadas: {r['regras_instanciadas']}  "
              f"disparadas: {r['regras_disparadas']}  "
              f"fatos derivados: {r['fatos_derivados']}  "
              f"tempo: {r['tempo_inferencia_s']:.3f}s")
        print("  exigencias:", ", ".join(r["exigencias"]))
        print("  playlist:", len(r["playlist"]), "faixas  "
              f"duracao: {r['duracao']}")
        print("  explicacao da primeira faixa:")
        for linha in r["explicacao"]:
            print("   " + linha)

    resultados = {
        "parametros": {
            "semente": SEMENTE,
            "n_execucoes": N_EXECUCOES,
            "limiares": LIMIARES,
            "alvos_numericos": {k: list(v) for k, v in ALVOS_NUMERICOS.items()},
        },
        "cenarios": {rotulo: {"fatos": sorted(c.fatos),
                              "duracao_alvo_min": c.duracao_alvo_s / 60.0,
                              "progressao": c.progressao}
                     for rotulo, c in CENARIOS.items()},
        "resultados": resumos,
    }

    caminho_json = os.path.join(DIR_RESULTADOS, "proj6_resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    rotulos = list(CENARIOS)
    caminho_csv = os.path.join(DIR_RESULTADOS, "proj6_tabela.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["cenario", "algoritmo", "cobertura",
                           "distancia_alvo", "tempo_inferencia_s"])
        for rotulo in rotulos:
            r = resumos[rotulo]
            m = r["metricas"]
            escritor.writerow([rotulo, "regras", f"{m['cobertura_regras']:.4f}",
                               f"{m['distancia_regras']:.4f}",
                               f"{r['tempo_inferencia_s']:.4f}"])
            escritor.writerow([rotulo, "gulosa", f"{m['cobertura_gulosa']:.4f}",
                               f"{m['distancia_gulosa']:.4f}", ""])
            escritor.writerow([rotulo, "aleatoria",
                               f"{m['cobertura_aleatoria']:.4f}",
                               f"{m['distancia_aleatoria']:.4f}", ""])

    plotar_comparacao(
        {chave: [resumos[r]["metricas"][chave] for r in rotulos]
         for chave in ("cobertura_regras", "cobertura_gulosa",
                       "cobertura_aleatoria")},
        ("cobertura_regras", "cobertura_gulosa", "cobertura_aleatoria"),
        os.path.join(DIR_FIGURAS, "proj6_fig1_cobertura.png"),
        "Cobertura das exigencias por cenario", "Cobertura media")
    plotar_comparacao(
        {chave: [resumos[r]["metricas"][chave] for r in rotulos]
         for chave in ("distancia_regras", "distancia_gulosa",
                       "distancia_aleatoria")},
        ("distancia_regras", "distancia_gulosa", "distancia_aleatoria"),
        os.path.join(DIR_FIGURAS, "proj6_fig2_comparacao.png"),
        "Distancia media aos alvos numericos por cenario",
        "Distancia media (menor e melhor)")

    print()
    print("Artefatos gravados em results/ e figures/.")


# ---------------------------------------------------------------------------
# 10. FIGURAS
# ---------------------------------------------------------------------------


def plotar_comparacao(series: dict[str, list[float]], ordem_series: tuple,
                      caminho: str, titulo: str, rotulo_y: str) -> None:
    """Desenha barras agrupadas por cenario para tres algoritmos."""
    rotulos = list(CENARIOS)
    x = list(range(len(rotulos)))
    n_series = len(ordem_series)
    largura = 0.75 / n_series
    nomes = {"regras": "Regras", "gulosa": "Gulosa", "aleatoria": "Aleatoria"}

    plt.figure(figsize=(8, 5))
    for i, chave in enumerate(ordem_series):
        deslocamento = (i - (n_series - 1) / 2) * largura
        rotulo = nomes.get(chave.replace("cobertura_", "")
                           .replace("distancia_", ""), chave)
        plt.bar([xi + deslocamento for xi in x], series[chave], largura,
                label=rotulo)
    plt.xticks(x, rotulos)
    plt.ylabel(rotulo_y)
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


if __name__ == "__main__":
    executar()
