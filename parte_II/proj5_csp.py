#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 5 - CSP: MONTAGEM DE GRADE HORARIA DA FGA
==================================================
Alocacao de turmas reais de Engenharia de Software (UnB Gama) a horarios e
salas usando Satisfacao de Restricoes, com cinco variantes de busca
comparadas entre si.

O problema e o mesmo que um coordenador enfrenta todo semestre: dado o
conjunto de turmas que serao ofertadas, cada uma com disciplina, professor,
numero de vagas e a quantidade de encontros semanais que precisa, decidir em
que bloco de horario e em que sala cada encontro acontece. Uma grade valida
nao pode colocar o mesmo professor em dois lugares ao mesmo tempo, nao pode
colocar duas turmas na mesma sala no mesmo horario e nao pode colocar uma
turma numa sala menor que o numero de vagas.

O problema e CSP por natureza: cada encontro de turma e uma variavel cujo
valor combina um bloco de horario com uma sala, e as exigencias acima sao
restricoes entre pares de variaveis. O que o torna interessante para a
disciplina e o tamanho dos dominios (centenas de pares bloco-sala por
variavel) combinado com a densidade das restricoes, um cenario classico em
que a escolha de heuristica e de inferencia separa a busca que termina em
milissegundos da busca que nao termina. Por isso o projeto compara cinco
variantes progressivas: backtracking puro, adicao de MRV, adicao de LCV,
adicao de forward checking e adicao de MAC com AC-3, medindo atribuicoes,
retornos e tempo de cada uma.

Alem das tres restricoes institucionais acima, o experimento testa duas
restricoes que parecem naturais mas que os dados reais desmentem: separar as
turmas da mesma disciplina em horarios diferentes e separar as disciplinas
do mesmo periodo do fluxo. A oferta real de 2026.2 mostra que as turmas dos
Projetos Integradores se encontram todas no mesmo horario, em salas
diferentes, e que o terceiro e o quarto periodos tem mais encontros que
blocos disponiveis na semana, o que torna a restricao de periodo infactivel
por simples contagem. O programa formaliza essa descoberta com uma funcao de
viabilidade e resolve a grade em tres cenarios.

Os dados de turmas, professores, horarios e vagas sao reais, extraidos da
oferta de Engenharia de Software 2026.2 no SIGAA e entregues em
data/grade_fga.csv no formato do arquivo modelo data/grade_fga_template.csv.
As salas sao modeladas com capacidades sinteticas, porque a lotacao fisica
das salas da faculdade nao e publica; as turmas e os professores sao reais.
Esta decisao esta registrada nos metadados dos resultados.

DEPENDENCIAS
  matplotlib (figuras). Nada alem da biblioteca padrao.

EXECUCAO
  python proj5_csp.py
  O programa monta a grade da instancia real em tres cenarios, roda o
  experimento de escala e grava tabelas em results/ e figuras em figures/.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# 1. CONFIGURACAO
# ===========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DADOS = os.path.join(BASE, "data")
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

# Arquivo com as turmas reais, no formato de data/grade_fga_template.csv.
ARQUIVO_TURMAS = os.path.join(DIR_DADOS, "grade_fga.csv")

# Semente para as instancias sinteticas do experimento de escala.
SEMENTE = 42

# Limites de seguranca da busca. Quando uma variante atinge o limite sem
# encontrar solucao, o experimento registra a variante como incompleta, em
# vez de travar a execucao.
MAX_NOS = 200_000
TEMPO_LIMITE_S = 120.0

# Salas sinteticas no formato (nome, capacidade). A capacidade real das salas
# da FGA nao e publica, entao os valores representam a ordem de grandeza do
# campus: dois auditorios, salas de aula de 60 a 120 lugares e laboratorios
# menores. As capacidades cobrem o intervalo de vagas das turmas reais, que
# vai de 16 a 131 lugares.
SALAS = [
    ("AUD-1", 140), ("AUD-2", 140),
    ("S1", 120), ("S2", 120), ("S3", 120), ("S4", 120),
    ("S5", 100), ("S6", 100), ("S7", 100), ("S8", 100),
    ("S9", 90), ("S10", 90),
    ("I1", 80), ("I2", 80), ("I3", 80), ("I4", 80),
    ("I5", 60), ("I6", 60), ("I7", 60), ("I8", 60),
    ("MOCAP", 100),
    ("LAB-1", 40), ("LAB-2", 40), ("LAB-3", 30), ("LAB-4", 24),
    ("LAB-5", 16),
]

# Dias uteis da UnB na notacao do SIGAA (2 = segunda, 7 = sabado) e os tres
# turnos. O sabado entra porque algumas turmas reais o usam.
DIAS = [2, 3, 4, 5, 6, 7]
TURNOS = ["M", "T", "N"]
BLOCOS = ["12", "34", "56"]

NOMES_DIA = {2: "seg", 3: "ter", 4: "qua", 5: "qui", 6: "sex", 7: "sab"}
HORARIOS = {
    ("M", "12"): "08:00-09:50", ("M", "34"): "10:00-11:50",
    ("M", "56"): "12:00-13:50",
    ("T", "12"): "14:00-15:50", ("T", "34"): "16:00-17:50",
    ("T", "56"): "18:00-19:50",
    ("N", "12"): "19:00-20:40", ("N", "34"): "20:50-22:30",
    ("N", "56"): "22:40-23:59",
}

# Tamanhos do experimento de escala, em numero de turmas sinteticas.
ESCALA_TURMAS = [20, 40, 60]


@dataclass(frozen=True)
class Turma:
    """Uma turma real do SIGAA com seus encontros semanais.

    O campo professor pode trazer mais de um nome separado por barra
    vertical, para turmas ministradas em parceria; cada nome entra na
    restricao de professor sem choque de forma independente.
    """

    codigo: str
    nome: str
    letra: str
    professor: str
    vagas: int
    encontros: tuple[tuple[int, str, str], ...]  # (dia, turno, bloco) por encontro
    periodo_fluxo: int | None


# ===========================================================================
# 2. LEITURA DOS DADOS
# ===========================================================================


def parsear_horarios_sigaa(texto: str) -> list[tuple[int, str, str]]:
    """Traduz a coluna de horarios em uma lista de (dia, turno, bloco).

    Os horarios chegam ja normalizados no CSV, um token por bloco, no
    formato 2M34 ou 4T12. Um grupo de quatro digitos como 1234 vira dois
    blocos, 12 e 34, e um grupo de dois digitos como 34 vira um bloco so.
    """
    encontros: list[tuple[int, str, str]] = []
    for token in texto.split():
        dia = int(token[0])
        turno = token[1]
        digitos = token[2:]
        if dia not in DIAS or turno not in TURNOS or len(digitos) % 2 != 0:
            raise ValueError("horario invalido no SIGAA: " + token)
        for i in range(0, len(digitos), 2):
            encontros.append((dia, turno, digitos[i:i + 2]))
    return encontros


def carregar_turmas(caminho: str) -> list[Turma]:
    """Le o CSV com as turmas e devolve a lista de Turma.

    O arquivo segue as colunas do modelo em data/grade_fga_template.csv:
    codigo, nome, letra, professor, vagas, horarios_sigaa e periodo_fluxo.
    O campo professor aceita nomes separados por barra vertical quando a
    turma e ministrada em parceria. O campo periodo_fluxo fica em branco
    para disciplinas fora do fluxo, como optativas.
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            "Arquivo de turmas nao encontrado em " + caminho + ". Exporte as "
            "turmas do SIGAA no formato de data/grade_fga_template.csv e "
            "salve como data/grade_fga.csv."
        )

    turmas: list[Turma] = []
    with open(caminho, encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            periodo = linha.get("periodo_fluxo", "").strip()
            turmas.append(Turma(
                codigo=linha["codigo_disciplina"].strip(),
                nome=linha["nome_disciplina"].strip(),
                letra=linha["turma"].strip(),
                professor=linha["professor"].strip(),
                vagas=int(linha["vagas"]) if linha["vagas"].strip() else 0,
                encontros=tuple(parsear_horarios_sigaa(linha["horarios_sigaa"])),
                periodo_fluxo=int(periodo) if periodo else None,
            ))
    return turmas


def gerar_turmas_sinteticas(n_turmas: int, semente: int) -> list[Turma]:
    """Gera uma instancia sintetica com n_turmas para o experimento de escala.

    As turmas sao agrupadas em periodos de poucas disciplinas, imitando a
    estrutura do fluxo real, e cada disciplina tem uma ou duas turmas. Os
    professores sao sorteados de um corpo docente reduzido para forcar a
    restricao de professor sem choque. Os horarios sorteados definem apenas
    quantos encontros cada turma tem, porque a busca recalcula todos os
    horarios do zero.
    """
    rng = random.Random(semente)
    corpo = [f"P{i:02d}" for i in range(1, max(4, n_turmas // 3 + 1))]
    turmas: list[Turma] = []
    disciplina = 0
    while len(turmas) < n_turmas:
        disciplina += 1
        periodo = (disciplina - 1) % 6 + 1
        n_repeticoes = rng.choice([1, 1, 2])
        for rep in range(n_repeticoes):
            if len(turmas) >= n_turmas:
                break
            vagas = rng.choice([25, 30, 40, 45, 50, 60])
            n_encontros = rng.choice([1, 2, 2])
            encontros = []
            for _ in range(n_encontros):
                encontros.append((rng.choice(DIAS), rng.choice(TURNOS),
                                  rng.choice(BLOCOS)))
            turmas.append(Turma(
                codigo=f"FGA9{disciplina:03d}",
                nome=f"DISCIPLINA {disciplina}",
                letra=chr(ord("A") + rep),
                professor=rng.choice(corpo),
                vagas=vagas,
                encontros=tuple(encontros),
                periodo_fluxo=periodo,
            ))
    return turmas


def bloco_para_texto(dia: int, turno: str, bloco: str) -> str:
    """Formata um bloco como aparece na grade: 'ter 10:00-11:50'."""
    return f"{NOMES_DIA[dia]} {HORARIOS[(turno, bloco)]}"


def _slots() -> list[tuple[int, str, str]]:
    return [(d, t, b) for d in DIAS for t in TURNOS for b in BLOCOS]


# Lista fixa de todos os blocos possiveis, pre-computada uma unica vez.
SLOTS = _slots()


def _relacao(csp: dict, var_a: str, var_b: str) -> str:
    """Devolve o rotulo da relacao entre duas variaveis, com chave canonica.

    Quando o par nao tem relacao forte registrada, a unica exigencia entre
    as duas variaveis e a unicidade da sala, representada pelo rotulo
    "sala".
    """
    par = (var_a, var_b) if var_a < var_b else (var_b, var_a)
    return csp["relacoes"].get(par, "sala")


# ===========================================================================
# 3. FORMALIZACAO DO CSP
# ===========================================================================
# Cada encontro de turma vira uma variavel com dominio igual aos pares
# (bloco, sala) cuja capacidade comporta as vagas da turma. O grafo de
# restricoes so guarda as relacoes fortes (professor, disciplina, periodo e
# encontros da mesma turma), que proibem o compartilhamento de bloco; a
# unicidade da sala e conferida na hora da atribuicao com um indice de salas
# ocupadas por bloco, porque uma aresta por par de variaveis nao acrescenta
# informacao a poda e deixaria o grafo denso demais.


def construir_csp(turmas: list[Turma],
                  incluir_disciplina: bool = True,
                  incluir_periodo: bool = True) -> dict:
    """Monta variaveis, dominios e relacoes a partir da lista de turmas.

    Os parametros ligam e desligam as duas restricoes nao institucionais. A
    restricao de professor, a unicidade da sala, a capacidade e a separacao
    dos encontros da mesma turma sao sempre ativas.
    """
    variaveis: list[str] = []
    dominio: dict[str, dict[tuple[int, str, str], list[tuple[str, int]]]] = {}
    info: dict[str, tuple[int, int]] = {}  # mapeia a variavel para turma e encontro
    relacoes: dict[tuple[str, str], str] = {}

    for i, turma in enumerate(turmas):
        salas_validas = [(nome, cap) for (nome, cap) in SALAS
                         if turma.vagas == 0 or cap >= turma.vagas]
        if not salas_validas:
            raise ValueError(
                f"turma {turma.codigo}-{turma.letra} com {turma.vagas} vagas "
                "nao cabe em nenhuma sala do modelo")
        for j, (dia, turno, bloco) in enumerate(turma.encontros):
            var = f"{turma.codigo}-{turma.letra}#{j}"
            variaveis.append(var)
            info[var] = (i, j)
            dominio[var] = {slot: list(salas_validas) for slot in SLOTS}

    def ligar(vars_: list[str], rotulo: str) -> None:
        for i in range(len(vars_)):
            for j in range(i + 1, len(vars_)):
                par = (vars_[i], vars_[j]) if vars_[i] < vars_[j] \
                    else (vars_[j], vars_[i])
                relacoes[par] = rotulo

    # Restricao por professor: variaveis do mesmo docente nao compartilham
    # bloco. Turmas em parceria repetem a regra para cada docente.
    por_professor: dict[str, list[str]] = {}
    for var in variaveis:
        i, _ = info[var]
        for nome in turmas[i].professor.split("|"):
            por_professor.setdefault(nome.strip(), []).append(var)
    for vars_ in por_professor.values():
        ligar(vars_, "professor")

    # Restricao opcional por disciplina: turmas diferentes da mesma
    # disciplina nao compartilham bloco.
    if incluir_disciplina:
        por_disciplina: dict[str, list[str]] = {}
        for var in variaveis:
            i, _ = info[var]
            por_disciplina.setdefault(turmas[i].codigo, []).append(var)
        for vars_ in por_disciplina.values():
            ligar(vars_, "disciplina")

    # Restricao opcional por periodo do fluxo: disciplinas do mesmo periodo
    # nao compartilham bloco.
    if incluir_periodo:
        por_periodo: dict[int, list[str]] = {}
        for var in variaveis:
            i, _ = info[var]
            if turmas[i].periodo_fluxo is not None:
                por_periodo.setdefault(turmas[i].periodo_fluxo, []).append(var)
        for vars_ in por_periodo.values():
            ligar(vars_, "periodo")

    # Os encontros da mesma turma sempre ocupam blocos diferentes.
    for var_a in variaveis:
        for var_b in variaveis:
            if var_a < var_b and info[var_a][0] == info[var_b][0]:
                relacoes[(var_a, var_b)] = "turma"

    vizinhos: dict[str, list[str]] = {v: [] for v in variaveis}
    for a, b in relacoes:
        vizinhos[a].append(b)
        vizinhos[b].append(a)

    return {
        "variaveis": variaveis,
        "dominio": dominio,
        "info": info,
        "turmas": turmas,
        "relacoes": relacoes,
        "vizinhos": vizinhos,
    }


def conflitam(csp: dict, var_a: str, val_a, var_b: str, val_b) -> bool:
    """Avalia a restricao binaria entre dois valores atribuidos.

    As relacoes fortes proibem o mesmo bloco; a relacao implicita de sala
    proibe o mesmo bloco apenas quando a sala coincide.
    """
    relacao = _relacao(csp, var_a, var_b)
    slot_a, slot_b = val_a[0], val_b[0]
    if relacao in ("professor", "disciplina", "periodo", "turma"):
        return slot_a == slot_b
    return slot_a == slot_b and val_a[1] == val_b[1]


def consistente(csp: dict, atribuicao: dict, salas_ocupadas: dict,
                var: str, val) -> bool:
    """Confere o valor contra as variaveis atribuidas e as salas ocupadas.

    A checagem forte percorre apenas os vizinhos do grafo de relacoes; a
    unicidade da sala usa o indice de salas por bloco, mantido junto com a
    atribuicao durante a busca.
    """
    slot, sala = val
    for vizinho in csp["vizinhos"][var]:
        if vizinho in atribuicao and conflitam(csp, var, val, vizinho,
                                               atribuicao[vizinho]):
            return False
    return sala not in salas_ocupadas.get(slot, ())


# ===========================================================================
# 4. INFERENCIA (PODA DE DOMINIOS)
# ===========================================================================
# As funcoes de inferencia mutam o dominio compartilhado e registram o
# estado anterior de cada faixa tocada, para o backtracking desfazer tudo ao
# voltar. O registro e um dicionario de variavel para um dicionario de bloco
# para a lista de salas anterior, tirada apenas na primeira mutacao da faixa.


def _registrar(estado: dict, dominio: dict, var: str, slot) -> None:
    if slot not in estado.setdefault(var, {}):
        estado[var][slot] = list(dominio[var].get(slot, []))


def remover_incompativeis(csp: dict, atribuicao: dict, dominio: dict,
                          var: str, val) -> tuple[dict, int]:
    """Remove do dominio de cada vizinho os valores que conflitam com val.

    Como o grafo so guarda relacoes fortes, a poda e sempre a remocao do
    bloco inteiro do dominio do vizinho. Devolve o estado das mutacoes para
    desfazer e o numero de valores podados.
    """
    estado: dict = {}
    podas = 0
    slot, _ = val
    for vizinho in csp["vizinhos"][var]:
        if vizinho in atribuicao:
            continue
        dom_vizinho = dominio[vizinho]
        if slot in dom_vizinho:
            _registrar(estado, dominio, vizinho, slot)
            podas += len(dom_vizinho[slot])
            del dom_vizinho[slot]
    return estado, podas


def revisar_arco(csp: dict, dominio: dict, xi: str, xj: str) -> tuple[dict, int]:
    """Remove de xi os valores sem par compativel no dominio de xj.

    Todas as arestas do grafo sao relacoes fortes, entao um valor sobrevive
    quando xj tem qualquer outro bloco disponivel. Devolve o estado e o
    numero de podas.
    """
    estado: dict = {}
    podas = 0
    for slot in list(dominio[xi]):
        compativel = len(dominio[xj]) > 1 or slot not in dominio[xj]
        if not compativel:
            _registrar(estado, dominio, xi, slot)
            podas += len(dominio[xi][slot])
            del dominio[xi][slot]
    return estado, podas


def desfazer(dominio: dict, estado: dict) -> None:
    """Restaura as faixas de dominio registradas no estado."""
    for var, faixas in estado.items():
        for slot, anterior in faixas.items():
            dominio[var][slot] = anterior


def inferencia_nenhuma(csp: dict, atribuicao: dict, dominio: dict,
                       var: str, val) -> tuple[dict, int]:
    """Nenhuma poda alem da conferencia de consistencia da atribuicao."""
    return {}, 0


def inferencia_forward_checking(csp: dict, atribuicao: dict, dominio: dict,
                                var: str, val) -> tuple[dict, int]:
    """Poda os vizinhos nao atribuidos imediatamente apos cada atribuicao."""
    return remover_incompativeis(csp, atribuicao, dominio, var, val)


def inferencia_mac(csp: dict, atribuicao: dict, dominio: dict,
                   var: str, val) -> tuple[dict, int]:
    """Mantem a arco consistencia (MAC) com AC-3 apos cada atribuicao.

    Alem da poda imediata nos vizinhos, propaga as remocoes entre os
    vizinhos entre si, porque um dominio reduzido pode tornar outros valores
    inviaveis sem que tenham conflito direto com a variavel atribuida.
    """
    estado, podas = remover_incompativeis(csp, atribuicao, dominio, var, val)

    fila = [(outro, vizinho)
            for vizinho in csp["vizinhos"][var] if vizinho not in atribuicao
            for outro in csp["vizinhos"][vizinho]
            if outro != var and outro not in atribuicao]

    while fila:
        xi, xj = fila.pop(0)
        estado_arco, podas_arco = revisar_arco(csp, dominio, xi, xj)
        for chave_var, faixas in estado_arco.items():
            for slot, anterior in faixas.items():
                if slot not in estado.setdefault(chave_var, {}):
                    estado[chave_var][slot] = anterior
        podas += podas_arco
        if not dominio[xi]:
            return estado, podas
        # Novos arcos so entram na fila quando a revisao removeu valores;
        # sem remocao nao ha o que propagar e re-enfileirar criaria ciclo.
        if podas_arco > 0:
            for xk in csp["vizinhos"][xi]:
                if xk != xj and xk not in atribuicao:
                    fila.append((xk, xi))
    return estado, podas


# ===========================================================================
# 5. BUSCA COM RETROCESSO
# ===========================================================================


def selecionar_primeira(atribuicao: dict, csp: dict, dominio: dict) -> str | None:
    """Seleciona a primeira variavel ainda nao atribuida na ordem original."""
    for var in csp["variaveis"]:
        if var not in atribuicao:
            return var
    return None


def selecionar_mrv(atribuicao: dict, csp: dict, dominio: dict) -> str | None:
    """Seleciona a variavel com o menor dominio restante (MRV).

    Em caso de empate fica a variavel mais conectada, que restringe mais
    variaveis no passo seguinte.
    """
    candidatas = [v for v in csp["variaveis"] if v not in atribuicao]
    if not candidatas:
        return None
    return min(candidatas,
               key=lambda v: (sum(len(salas) for salas in dominio[v].values()),
                              -len(csp["vizinhos"][v])))


def ordenar_natural(csp: dict, var: str, dominio: dict):
    """Devolve os valores do dominio na ordem em que foram gerados."""
    valores = []
    for slot in SLOTS:
        for sala in dominio[var].get(slot, []):
            valores.append((slot, sala))
    return valores


def ordenar_lcv(csp: dict, var: str, dominio: dict):
    """Ordena os valores pelo efeito de poda sobre os vizinhos (LCV).

    Um valor que elimina menos opcoes dos vizinhos e preferido, deixando
    opcoes em aberto para as variaveis futuras.
    """
    def impacto(val):
        slot, _ = val
        total = 0
        for vizinho in csp["vizinhos"][var]:
            if slot in dominio[vizinho]:
                total += len(dominio[vizinho][slot])
        return total

    return sorted(ordenar_natural(csp, var, dominio), key=impacto)


def backtracking(csp: dict, selecionar, ordenar, inferir,
                 max_nos: int = MAX_NOS,
                 tempo_limite: float = TEMPO_LIMITE_S) -> dict:
    """Busca em profundidade com retrocesso sobre a atribuicao parcial.

    Cada no escolhe uma variavel pela funcao de selecao, tenta os valores na
    ordem da funcao de ordenacao e aplica a inferencia escolhida antes de
    descer, desfazendo a poda na volta. As metricas contam nos visitados,
    atribuicoes feitas, retornos e valores podados, e a busca para ao atingir
    o limite de nos ou de tempo, caso em que devolve a flag completo em
    falso.
    """
    estatisticas = {"nos": 0, "atribuicoes": 0, "retornos": 0, "podas": 0}
    t0 = time.perf_counter()
    salas_ocupadas: dict[tuple[int, str, str], set[tuple[str, int]]] = {}

    def buscar(atribuicao: dict, dominio: dict):
        if estatisticas["nos"] >= max_nos or time.perf_counter() - t0 > tempo_limite:
            return None
        estatisticas["nos"] += 1

        var = selecionar(atribuicao, csp, dominio)
        if var is None:
            return dict(atribuicao)

        for val in ordenar(csp, var, dominio):
            if not consistente(csp, atribuicao, salas_ocupadas, var, val):
                estatisticas["retornos"] += 1
                continue
            estatisticas["atribuicoes"] += 1
            estado, podas = inferir(csp, atribuicao, dominio, var, val)
            estatisticas["podas"] += podas
            falhou = any(not dominio[v]
                         for v in csp["vizinhos"][var] if v not in atribuicao)
            if falhou:
                estatisticas["retornos"] += 1
                desfazer(dominio, estado)
                continue
            atribuicao[var] = val
            salas_ocupadas.setdefault(val[0], set()).add(val[1])
            resultado = buscar(atribuicao, dominio)
            if resultado is not None:
                return resultado
            salas_ocupadas[val[0]].discard(val[1])
            del atribuicao[var]
            estatisticas["retornos"] += 1
            desfazer(dominio, estado)

        return None

    solucao = buscar({}, csp["dominio"])
    estatisticas["tempo"] = time.perf_counter() - t0
    estatisticas["completo"] = solucao is not None
    estatisticas["infactivel"] = (solucao is None
                                  and estatisticas["nos"] < max_nos
                                  and estatisticas["tempo"] < tempo_limite)
    return {"solucao": solucao or {}, **estatisticas}


VARIANTES = [
    ("backtracking puro", selecionar_primeira, ordenar_natural, inferencia_nenhuma),
    ("MRV", selecionar_mrv, ordenar_natural, inferencia_nenhuma),
    ("MRV+LCV", selecionar_mrv, ordenar_lcv, inferencia_nenhuma),
    ("MRV+LCV+FC", selecionar_mrv, ordenar_lcv, inferencia_forward_checking),
    ("MRV+LCV+MAC", selecionar_mrv, ordenar_lcv, inferencia_mac),
]


# ===========================================================================
# 6. VERIFICACAO INDEPENDENTE DA SOLUCAO
# ===========================================================================


def verificar_solucao(csp: dict, solucao: dict) -> list[str]:
    """Conferencia externa da solucao, sem usar a busca.

    Percorre todos os pares de variaveis e lista cada violacao de restricao,
    alem de conferir capacidade de sala e bloco valido. Devolve a lista de
    violacoes, que precisa estar vazia para a solucao ser aceita.
    """
    violacoes = []
    turmas = csp["turmas"]
    info = csp["info"]
    for var, val in solucao.items():
        i, _ = info[var]
        sala = val[1]
        if turmas[i].vagas > 0 and sala[1] < turmas[i].vagas:
            violacoes.append(f"{var}: sala {sala[0]} menor que as vagas")
    for var_a in solucao:
        for var_b in solucao:
            if var_a < var_b and conflitam(csp, var_a, solucao[var_a],
                                           var_b, solucao[var_b]):
                violacoes.append(f"{var_a} x {var_b} conflitam")
    return violacoes


# ===========================================================================
# 7. VIABILIDADE POR CONTAGEM
# ===========================================================================


def verificar_viabilidade_por_contagem(turmas: list[Turma],
                                       incluir_disciplina: bool = True,
                                       incluir_periodo: bool = True) -> list[dict]:
    """Prova a inviabilidade de restricoes por argumento de contagem.

    As restricoes de disciplina e de periodo exigem blocos todos diferentes
    dentro do grupo. Se um grupo tem mais encontros que os blocos
    disponiveis na semana, nenhuma grade satisfaz a restricao, por principio
    da casa dos pombos. Devolve a lista de grupos inviaveis com o numero de
    encontros e o limite.
    """
    inviaveis = []
    total_blocos = len(SLOTS)
    if incluir_disciplina:
        por_disciplina: dict[str, int] = {}
        for turma in turmas:
            por_disciplina[turma.codigo] = \
                por_disciplina.get(turma.codigo, 0) + len(turma.encontros)
        for codigo, n in por_disciplina.items():
            if n > total_blocos:
                inviaveis.append({"tipo": "disciplina", "chave": codigo,
                                  "encontros": n, "blocos": total_blocos})
    if incluir_periodo:
        por_periodo: dict[int, int] = {}
        for turma in turmas:
            if turma.periodo_fluxo is not None:
                por_periodo[turma.periodo_fluxo] = \
                    por_periodo.get(turma.periodo_fluxo, 0) + len(turma.encontros)
        for periodo, n in por_periodo.items():
            if n > total_blocos:
                inviaveis.append({"tipo": "periodo", "chave": str(periodo),
                                  "encontros": n, "blocos": total_blocos})
    return inviaveis


# ===========================================================================
# 8. EXPERIMENTO
# ===========================================================================


def _rodar_variante(nome: str, selecionar, ordenar, inferir, csp: dict) -> dict:
    """Executa uma variante e resume suas metricas."""
    resultado = backtracking(csp, selecionar, ordenar, inferir)
    return {
        "variante": nome,
        "completo": resultado["completo"],
        "infactivel": resultado["infactivel"],
        "nos": resultado["nos"],
        "atribuicoes": resultado["atribuicoes"],
        "retornos": resultado["retornos"],
        "podas": resultado["podas"],
        "tempo": round(resultado["tempo"], 4),
        "solucao": resultado["solucao"] if resultado["completo"] else None,
    }


def _solucao_para_json(solucao: dict) -> dict:
    return {var: [list(val[0]), list(val[1])] for var, val in solucao.items()}


def _experimento_cenario(nome: str, turmas: list[Turma], csp: dict) -> dict:
    """Roda as cinco variantes em um cenario e verifica a melhor solucao."""
    resultados = {}
    solucoes = {}
    for variante, selecionar, ordenar, inferir in VARIANTES:
        print(f"    [{variante}] ", end="", flush=True)
        t0 = time.perf_counter()
        r = _rodar_variante(variante, selecionar, ordenar, inferir, csp)
        print(f"completo={r['completo']} infactivel={r['infactivel']} "
              f"nos={r['nos']} retornos={r['retornos']} "
              f"({time.perf_counter() - t0:.2f}s)", flush=True)
        resultados[variante] = r
        if r["solucao"]:
            solucoes[variante] = r["solucao"]

    verificacao = []
    solucao_ref = None
    if solucoes:
        solucao_ref = solucoes.get("MRV+LCV+MAC") or next(iter(solucoes.values()))
        verificacao = verificar_solucao(csp, solucao_ref)

    return {
        "cenario": nome,
        "turmas": len(turmas),
        "variaveis": len(csp["variaveis"]),
        "resultados": {v: {k: val for k, val in r.items() if k != "solucao"}
                       for v, r in resultados.items()},
        "verificacao": verificacao,
        "solucao_referencia": _solucao_para_json(solucao_ref) if solucao_ref else None,
    }


def experimento_instancia_real(turmas: list[Turma]) -> dict:
    """Roda os cenarios da instancia real com restricoes crescentes."""
    cenarios = {}

    print("  cenario A: professor, sala e capacidade", flush=True)
    csp_a = construir_csp(turmas, incluir_disciplina=False, incluir_periodo=False)
    cenarios["A_institucional"] = _experimento_cenario(
        "A_institucional", turmas, csp_a)

    print("  cenario B: acrescenta separacao das turmas da disciplina", flush=True)
    csp_b = construir_csp(turmas, incluir_disciplina=True, incluir_periodo=False)
    cenarios["B_disciplina"] = _experimento_cenario("B_disciplina", turmas, csp_b)

    print("  cenario C: acrescenta o periodo do fluxo", flush=True)
    inviaveis = verificar_viabilidade_por_contagem(
        turmas, incluir_disciplina=True, incluir_periodo=True)
    cenarios["C_periodo"] = {
        "cenario": "C_periodo",
        "turmas": len(turmas),
        "variaveis": None,
        "resultados": None,
        "verificacao": [],
        "solucao_referencia": None,
        "inviavel_por_contagem": inviaveis,
    }
    if inviaveis:
        for item in inviaveis:
            print(f"    inviavel por contagem: {item}", flush=True)
    else:
        csp_c = construir_csp(turmas, incluir_disciplina=True, incluir_periodo=True)
        cenarios["C_periodo"] = _experimento_cenario("C_periodo", turmas, csp_c)

    return {"salas": SALAS, "cenarios": cenarios}


def experimento_escala() -> list[dict]:
    """Mede as variantes em instancias sinteticas de 20, 40 e 60 turmas."""
    linhas = []
    for n in ESCALA_TURMAS:
        turmas = gerar_turmas_sinteticas(n, SEMENTE)
        csp = construir_csp(turmas)
        print(f"  escala {n} turmas ({len(csp['variaveis'])} variaveis):",
              flush=True)
        for nome, selecionar, ordenar, inferir in VARIANTES:
            r = _rodar_variante(nome, selecionar, ordenar, inferir, csp)
            linhas.append({"turmas": n,
                           **{k: v for k, v in r.items() if k != "solucao"}})
            print(f"    [{nome}] completo={r['completo']} nos={r['nos']} "
                  f"retornos={r['retornos']} ({r['tempo']}s)", flush=True)
    return linhas


def imprimir_grade(turmas: list[Turma], solucao: dict) -> None:
    """Imprime a grade final ordenada por bloco, no formato de quadro."""
    csp = construir_csp(turmas, incluir_disciplina=False, incluir_periodo=False)
    info = csp["info"]
    linhas = []
    for var, val in solucao.items():
        i, _ = info[var]
        turma = turmas[i]
        slot, sala = val
        linhas.append((slot, sala[0], turma.nome, turma.letra, turma.professor))
    linhas.sort(key=lambda l: (DIAS.index(l[0][0]), TURNOS.index(l[0][1]),
                               BLOCOS.index(l[0][2])))

    print()
    print("=" * 76)
    print("                       GRADE HORARIA ENCONTRADA")
    print("=" * 76)
    print()
    for slot, sala, nome, letra, professor in linhas:
        rotulo = bloco_para_texto(*slot)
        docente = professor.replace("|", " / ")
        print(f"{rotulo:<17} {sala:<7} {nome} - {letra} ({docente})")
    print()


def executar() -> None:
    """Fluxo completo: cenarios reais, escala, artefatos e grade."""
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)

    turmas = carregar_turmas(ARQUIVO_TURMAS)
    print(f"Instancia real: {len(turmas)} turmas carregadas de "
          f"{os.path.basename(ARQUIVO_TURMAS)}.")
    print("Rodando os cenarios da instancia real...")
    resumo_real = experimento_instancia_real(turmas)

    print("Rodando o experimento de escala...")
    escala = experimento_escala()

    solucao_a = resumo_real["cenarios"]["A_institucional"]["solucao_referencia"]
    if solucao_a:
        solucao = {}
        for var, (slot, sala) in solucao_a.items():
            solucao[var] = (tuple(slot), tuple(sala))
        imprimir_grade(turmas, solucao)

    caminho_json = os.path.join(DIR_RESULTADOS, "proj5_resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump({"instancia_real": resumo_real, "escala": escala}, fh,
                  ensure_ascii=False, indent=2)

    caminho_csv = os.path.join(DIR_RESULTADOS, "proj5_tabela.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["cenario", "variante", "completo", "infactivel",
                           "nos", "atribuicoes", "retornos", "podas", "tempo"])
        for nome_cenario, cenario in resumo_real["cenarios"].items():
            for variante, r in (cenario.get("resultados") or {}).items():
                escritor.writerow([nome_cenario, variante, r["completo"],
                                   r["infactivel"], r["nos"], r["atribuicoes"],
                                   r["retornos"], r["podas"], r["tempo"]])

    caminho_escala = os.path.join(DIR_RESULTADOS, "proj5_escala.csv")
    with open(caminho_escala, "w", newline="", encoding="utf-8") as fh:
        campos = ["turmas", "variante", "completo", "infactivel", "nos",
                  "atribuicoes", "retornos", "podas", "tempo"]
        escritor = csv.DictWriter(fh, fieldnames=campos)
        escritor.writeheader()
        for linha in escala:
            escritor.writerow(linha)

    plotar_variantes(resumo_real,
                     os.path.join(DIR_FIGURAS, "proj5_fig1_variantes.png"))
    plotar_escala(escala,
                  os.path.join(DIR_FIGURAS, "proj5_fig2_escala.png"))

    print("Artefatos gravados em results/ e figures/.")


# ===========================================================================
# 9. FIGURAS
# ===========================================================================


def plotar_variantes(resumo: dict, caminho: str) -> None:
    """Barras com retornos (escala log) de cada variante no cenario A."""
    cenario = resumo["cenarios"]["A_institucional"]
    resultados = cenario["resultados"]
    nomes = list(resultados)
    retornos = [resultados[n]["retornos"] for n in nomes]
    completos = [resultados[n]["completo"] for n in nomes]

    plt.figure(figsize=(9, 5))
    cores = ["#4c72b0" if c else "#c44e52" for c in completos]
    plt.bar(nomes, [max(r, 1) for r in retornos], color=cores)
    plt.yscale("log")
    plt.ylabel("Retornos (escala log)")
    plt.title("Esforco de cada variante na instancia real (cenario A)")
    plt.xticks(rotation=15, ha="right")
    for i, (r, c) in enumerate(zip(retornos, completos)):
        plt.text(i, max(r, 1) * 1.15, str(r) if c else "incompleto",
                 ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def plotar_escala(escala: list[dict], caminho: str) -> None:
    """Nos visitados (escala log) contra o numero de turmas, por variante."""
    plt.figure(figsize=(9, 5))
    por_variante: dict[str, dict[int, int]] = {}
    for linha in escala:
        por_variante.setdefault(linha["variante"], {})[linha["turmas"]] = \
            linha["nos"]
    for nome, pontos in por_variante.items():
        xs = sorted(pontos)
        ys = [pontos[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=nome, linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Turmas na instancia")
    plt.ylabel("Nos visitados (escala log)")
    plt.title("Crescimento do esforco de busca com o tamanho da instancia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


if __name__ == "__main__":
    executar()
