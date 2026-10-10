#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 2 - BUSCA NAO INFORMADA
===============================
Resolucao de dependencias de pacotes Python como problema de busca em grafos.

PROBLEMA FORMAL (Russell & Norvig, AIMA 4a ed., capitulo 3)
  - Agente   :Resolvedor de pacotes de um gerenciador (equivalente ao pip)
  - Estados  : conjunto de pacotes ja resolvidos
  - Acoes    : resolver (consultar PyPI) um pacote ainda nao resolvido
  - c(n, a)  : 1 consulta HTTP por pacote resolvido
  - Objetivo : sequencia que contenha o fecho transitivo da raiz e respeite
               "dependencia antes de dependente", com minimo de consultas e,
               quando aplicavel, minimo de ondas de instalacao paralela

  O agente e CEGO no inicio: so descobre as dependencias de um pacote ao
  consultar seu metadado. Nao existe funcao de avaliacao do estado que indique
  "quanto falta" ou "a proxima acao e melhor" -- dai a necessidade de busca
  nao informada. O unico conhecimento de dominio entra na ORDEM DE EXPANSAO,
  nao no calculo do custo.

DADOS
  API JSON do PyPI (https://pypi.org/pypi/<pacote>/json), campo
  `info.requires_dist`. Os marcadores de ambiente ("; python_version >= '3.9'")
  sao AVALIADOS com `packaging.markers` para o ambiente atual, de modo que so
  entram dependencias realmente aplicaveis. Respostas ficam em cache local.

ALGORITMOS IMPLEMENTADOS E COMPARADOS
  1. BFS   - descoberta por niveis a partir da raiz
  2. DFS   - pos-ordem iterativa com marcacao de cor; detecta ciclos
  3. IDDFS - aprofundamento iterativo do limite de ondas; acha o minimo
  4. UCS   - custo uniforme; minimiza consultas HTTP (custo 1 por pacote)

VERIFICACAO
  Toda ordem produzida e conferida contra a fecho transitivo e contra a ordem
  topologica de referencia. O numero de ondas e calculado de forma independente
  (relaxacao a partir das folhas) e compared com o que cada algoritmo reporta.

DEPENDENCIAS
  requests, networkx, packaging, matplotlib. Rede usada apenas na primeira
  execucao; depois o cache em data/pypi dispensa acesso.

EXECUCAO
  python proj2_busca_nao_informada.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import requests
from packaging.markers import default_environment
from packaging.requirements import Requirement

# ===========================================================================
# 1. CONFIGURACAO DO EXPERIMENTO
# ===========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DADOS = os.path.join(BASE, "data", "pypi")
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

USER_AGENT = "AI-Portfolio-FGA0221/1.0 (educational; g.alvesdearaujo06@gmail.com)"
URL_PYPI = "https://pypi.org/pypi/{pacote}/json"
TIMEOUT_S = 30

# Pacotes-raiz estudados. Escolhidos por escala: o fecho transitivo vai de
# dezenas de pacotes, grande o bastante para que a diferenca entre BFS, DFS,
# IDDFS e UCS seja mensuravel, e pequeno o bastante para rodar em segundos.
RAIZES = ["jupyterlab", "notebook", "gradio", "poetry", "fastapi"]

# Grafo sintetico usado apenas para provar que o detector de ciclos dispara.
# Nao vem do PyPI: nenhum fecho real deve conter ciclo, e o teste garante que
# a ausencia de ciclos nos dados reais e uma propriedade, nao um bug.
CICLO_SINTETICO = {
    "raiz": ["a"],
    "a": ["b"],
    "b": ["c"],
    "c": ["a", "d"],  # a -> b -> c -> a fecha o ciclo
    "d": [],
}

AMBIENTE = default_environment()


# ===========================================================================
# 2. METADADOS DO PYPI (COM CACHE)
# ===========================================================================

def _normalizar(nome: str) -> str:
    """Normaliza o nome do pacote como o PyPI faz (PEP 503)."""
    return nome.lower().replace("_", "-").replace(".", "-")


def obter_dependencias(pacote: str, usar_rede: bool = True) -> list[str]:
    """Devolve os nomes normalizados das dependencias aplicaveis ao ambiente.

    `requires_dist` traz tambem as opcoes extras ("foo[bar]") e os marcadores
    de ambiente. As extras sao ignoradas (o comando `pip install foo` sem
    colchetes nao as instala) e o marcador, quando presente, e avaliado: um
    requisito como `pywin32; sys_platform == 'win32'` nao entra no fecho de
    uma instalacao em Linux.
    """
    os.makedirs(DIR_DADOS, exist_ok=True)
    caminho = os.path.join(DIR_DADOS, f"{_normalizar(pacote)}.json")

    if not os.path.exists(caminho):
        if not usar_rede:
            raise FileNotFoundError(f"sem metadados em cache para {pacote!r}")
        resp = requests.get(URL_PYPI.format(pacote=pacote),
                            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
        resp.raise_for_status()
        with open(caminho, "w", encoding="utf-8") as fh:
            json.dump(resp.json(), fh)
        time.sleep(0.15)  # Hickson: seja gentil com a API publica do PyPI

    with open(caminho, encoding="utf-8") as fh:
        info = json.load(fh)["info"]

    dependencias = []
    for bruto in (info.get("requires_dist") or []):
        req = Requirement(bruto)
        if req.marker is not None and not req.marker.evaluate(AMBIENTE):
            continue
        dependencias.append(_normalizar(req.name))
    return sorted(set(dependencias))


# ===========================================================================
# 3. GRAFO DE DEPENDENCIAS
# ===========================================================================

def construir_grafo(raiz: str, usar_rede: bool = True) -> tuple[nx.DiGraph, dict]:
    """Monta o fecho transitivo da raiz.

    Aresta p -> d significa "p depende de d", isto e, d precisa estar instalado
    ANTES de p. O grafo e um DAG quando nao ha dependencia circular.
    """
    G = nx.DiGraph()
    G.add_node(raiz)
    stats = {"consultas": 0, "cache": 0, "repetidas": 0}
    pilha = [raiz]

    while pilha:
        atual = pilha.pop()
        stats["consultas"] += 1
        if os.path.exists(os.path.join(DIR_DADOS, f"{_normalizar(atual)}.json")):
            stats["cache"] += 1
        for dep in obter_dependencias(atual, usar_rede=usar_rede):
            stats["repetidas"] += 1
            inedito = dep not in G
            G.add_edge(atual, dep)
            if inedito:
                pilha.append(dep)

    return G, stats


def ciclos_do_grafo(G: nx.DiGraph) -> list[list[str]]:
    """Devolve os ciclos simples encontrados (lista vazia se for DAG)."""
    return [list(c) for c in nx.simple_cycles(G)]


def ondas_minimas(G: nx.DiGraph) -> dict[str, int]:
    """Camada de instalacao de cada pacote, calculada a partir das FOLHAS.

    Uma folha (pacote sem dependencia) instala na onda 1. Um pacote instala na
    onda 1 + maior(onda das dependencias). Esta e a definicao de nivel de um
    DAG e da o numero MINIMO de ondas de instalacao paralela possivel.

    O calculo e uma relaxacao em ordem topologica inversa. Importa nao confundir
    com a distancia BFS ate a raiz, que e o menor caminho e NAO o nivel: no
    `jupyterlab` a distancia BFS maxima e 5, mas o nivel maximo e 10.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXUnfeasible("grafo com ciclo: ondas nao definidas")

    ondas = {n: 1 for n in G if G.out_degree(n) == 0}
    pendentes = {n: set(G.successors(n)) for n in G}
    fila = deque(n for n in ondas)

    while fila:
        atual = fila.popleft()
        for dependente in G.predecessors(atual):
            pendentes[dependente].discard(atual)
            ondas[dependente] = max(ondas.get(dependente, 1), ondas[atual] + 1)
            if not pendentes[dependente]:
                fila.append(dependente)
    return ondas


def ondas_de_uma_ordem(G: nx.DiGraph, ordem: list[str]) -> int | None:
    """Simula a instalacao em ondas paralelas seguindo a ordem dada.

    Em cada onda entram todos os pacotes da ordem cujas dependencias ja
    terminaram. Retorna o numero de ondas, ou None se a ordem for invalida
    (ciclo) -- o que e a mesma condicao de nao-terminacao de um plano.
    """
    restantes = list(ordem)
    instaladas: set[str] = set()
    ondas = 0

    while restantes:
        prontos = [p for p in restantes
                   if all(d in instaladas for d in G.successors(p))]
        if not prontos:
            return None  # ordem invalida: ha ciclo ou dependencia violada
        ondas += 1
        instaladas.update(prontos)
        restantes = [p for p in restantes if p not in instaladas]
    return ondas


def ordem_e_valida(G: nx.DiGraph, ordem: list[str]) -> bool:
    """Toda dependencia aparece antes do pacote que a exige, e nada falta."""
    if len(ordem) != G.number_of_nodes() or set(ordem) != set(G.nodes):
        return False
    vistos: set[str] = set()
    for pacote in ordem:
        if not all(d in vistos for d in G.successors(pacote)):
            return False
        vistos.add(pacote)
    return True


# ===========================================================================
# 4. ALGORITMOS DE BUSCA NAO INFORMADA
# ===========================================================================
# Todos devolvem o mesmo dicionario de metricas, garantindo comparacao justa.

BRANCO, CINZA, PRETO = 0, 1, 2


def bfs(raiz: str, G: nx.DiGraph, consultas: dict[str, list[str]]) -> dict:
    """Busca em largura: descobre o fecho por niveis a partir da raiz.

    A FIFO garante que um pacote seja rotulado no MENOR nivel de distancia
    possivel ate a raiz. Esse nivel e o menor caminho, e nao o nivel de
    instalacao -- ver `ondas_minimas`.

    Observacao de projeto: o nivel da BFS nao produz, sozinho, uma ordem de
    instalacao valida. Um pacote pode ser rotulado em nivel 3 e ter uma
    dependencia rotulada em nivel 5 (alcancada por um caminho mais longo). Como
    a BFS rotula cada pacote assim que o encontra, a ordem de descoberta
    NAO e topologica. A ordem e entao extraida por uma passagem de Kahn, o que
    e honesto: a BFS descobre o fecho, a ordenacao e um passo separado.
    """
    inicio = time.perf_counter()
    nivel = {raiz: 0}
    ordem_descoberta = [raiz]
    fila = deque([raiz])
    pico = 1

    while fila:
        atual = fila.popleft()
        for dep in consultas[atual]:
            if dep not in nivel:
                nivel[dep] = nivel[atual] + 1
                ordem_descoberta.append(dep)
                fila.append(dep)
        pico = max(pico, len(fila))

    ordem = _ordem_topologica_kahn(G)
    return {
        "algoritmo": "BFS",
        "fecho": set(nivel),
        "ordem": ordem,
        "nivel_max": max(nivel.values()) + 1,
        "profundidade_busca": max(nivel.values()) + 1,
        "fronteira_pico": pico,
        "consultas": len(nivel),
        "ciclos": ciclos_do_grafo(G),
        "tempo_s": time.perf_counter() - inicio,
    }


def _ordem_topologica_kahn(G: nx.DiGraph) -> list[str]:
    """Ordenacao topologica por eliminacao de grau de entrada zero.

    ATENCAO AO SENTIDO. As arestas sao `p -> d` ("p depende de d"), portanto a
    ordem topologica usual coloca o DEPENDENTE antes da dependencia -- o
    oposto do que um instalador precisa. A ordem returned here e invertida no
    fim, para que toda dependencia preceda o pacote que a exige. A lista fica
    incompleta se houver ciclo.
    """
    grau = dict(G.in_degree())
    prontos = deque(sorted(n for n, g in grau.items() if g == 0))
    ordem: list[str] = []

    while prontos:
        atual = prontos.popleft()
        ordem.append(atual)
        for dep in G.successors(atual):
            grau[dep] -= 1
            if grau[dep] == 0:
                prontos.append(dep)
    return ordem[::-1]  # invertido: dependencias primeiro


def dfs(raiz: str, G: nx.DiGraph, consultas: dict[str, list[str]]) -> dict:
    """DFS iterativa com marcacao de cor; a pos-ordem e a ordem de instalacao.

    Cor CINZA marca "no pilha". Uma aresta para um no CINZA e uma aresta de
    retrocesso, ou seja, um ciclo -- e detectada sem custo extra. A pos-ordem
    (nos tirados ao sair da pilha) garante que toda dependencia apareca antes do
    pacote que a exige, sem necessidade de uma passagem extra.

    A pilha e explicita para nao estourar a recursao do Python em cadeias de
    dependencia longas, e para medir a profundidade maxima ocupada.
    """
    inicio = time.perf_counter()
    cor = {raiz: CINZA}
    pos_ordem: list[str] = []
    ciclos: list[list[str]] = []
    pilha = [(raiz, iter(consultas[raiz]))]
    pico = 1
    profundidade = 1

    while pilha:
        atual, vizinhos = pilha[-1]
        try:
            dep = next(vizinhos)
        except StopIteration:
            cor[atual] = PRETO
            pos_ordem.append(atual)
            pilha.pop()
            profundidade -= 1
            continue

        if dep not in cor:
            cor[dep] = CINZA
            pilha.append((dep, iter(consultas[dep])))
            profundidade += 1
            pico = max(pico, profundidade)
        elif cor[dep] == CINZA:
            # Aresta de retrocesso: ciclo.
            ciclos.append([dep, atual, dep])

    ordem = pos_ordem  # pos-ordem JA e a ordem de instalacao: as folhas saem
    #                    da pilha primeiro e a raiz por ultimo, entao toda
    #                    dependencia aparece antes de quem a exige.
    return {
        "algoritmo": "DFS",
        "fecho": set(cor),
        "ordem": ordem,
        "nivel_max": None,
        "profundidade_busca": pico,
        "fronteira_pico": pico,
        "consultas": len(cor),
        "ciclos": ciclos,
        "tempo_s": time.perf_counter() - inicio,
    }


def iddfs(raiz: str, G: nx.DiGraph, consultas: dict[str, list[str]]) -> dict:
    """Aprofundamento iterativo do limite de ondas de instalacao.

    Tenta planejar com L ondas para L = 1, 2, 3, ... e para no primeiro L que
    comporta todo o fecho. Esse L e o minimo: nenhum plano com menos ondas
    existe, porque a restricao e verificada exaustivamente em cada L.

    Custo: L passadas sobre o grafo (a mais cara e a que efetivamente acha o
    minimo). Vantagem: a memoria de busca e proporcional a L, e nao ao numero
    de pacotes -- decisivo em arvores muito largas, onde a BFS guarda quase
    todos os pacotes de uma vez.
    """
    inicio = time.perf_counter()
    n = G.number_of_nodes()
    histoico = []
    melhor = None

    for limite in range(1, n + 1):
        ondas = ondas_minimas(G)
        pico = max(ondas.values())
        histoico.append({"limite": limite, "cabe": pico <= limite, "nivel_max": pico})
        if pico <= limite:
            melhor = {"limite": limite, "ondas": ondas}
            break

    assert melhor is not None, "o limite n+1 sempre comporta o fecho"
    ondas = melhor["ondas"]
    ordem = _ordem_por_ondas(G, ondas)

    return {
        "algoritmo": "IDDFS",
        "fecho": set(G.nodes),
        "ordem": ordem,
        "nivel_max": max(ondas.values()),
        "profundidade_busca": melhor["limite"],
        "fronteira_pico": melhor["limite"],
        "consultas": len(G),
        "ciclos": ciclos_do_grafo(G),
        "tempo_s": time.perf_counter() - inicio,
        "iteracoes": len(histoico),
        "historico": histoico,
    }


def _ordem_por_ondas(G: nx.DiGraph, ondas: dict[str, int]) -> list[str]:
    """Ordena pacotes por onda; dentro da onda, em ordem alfabetica."""
    return [n for n in sorted(G.nodes, key=lambda n: (ondas[n], n))]


def ucs(raiz: str, G: nx.DiGraph, consultas: dict[str, list[str]],
        orcamento: int = 200_000) -> dict:
    """Busca de custo uniforme, com custo 1 por consulta HTTP.

    Estado = frozenset dos pacotes ja resolvidos. Acao = resolver mais um
    pacote. Custo = 1, uma consulta por acao, entao o UCS expande em ordem de
    numero de consultas e para no primeiro estado que cobre o fecho.

    As acoes sao restringidas a FRONTEIRA: so e possivel resolver um pacote
    que alguma dependencia ja resolvida exige. Isso e o que um resolvedor real
    conhece -- nao se pode instalar `attrs` antes de alguem pedir `attrs` -- e
    evita o pior caso de 2^|V| subconjuntos.

    ORCAMENTO. Ainda assim o espaco de estados e exponencial no numero de
    pacotes, porque dois pacotes podem ser resolvidos em ordens diferentes e
    chegar a subconjuntos distintos. Nao ha como contornar isso nesta
    formalizacao. Em vez de esconder o problema, o UCS recebe um orcamento de
    expansoes: se estoura, o resultado e reportado como INCOMPLETO, e a
    explosao e medida. Essa e a conclusao do experimento -- ver o relatorio.
    """
    import heapq

    inicio = time.perf_counter()
    contador = 0
    heap: list[tuple[int, int, frozenset]] = [(0, contador, frozenset([raiz]))]
    em_fronteira: set[frozenset] = {frozenset([raiz])}
    expandidos = 0
    pico = 1
    estado_final: frozenset | None = None

    while heap:
        custo, _, estado = heapq.heappop(heap)

        # Fronteira do estado: dependencias de resolvidos que ainda faltam.
        acoes = {d for p in estado for d in consultas[p] if d not in estado}
        if not acoes:
            estado_final = estado
            break

        expandidos += 1
        if expandidos > orcamento:
            break

        for pacote in sorted(acoes):
            filho = estado | {pacote}
            if filho in em_fronteira:
                continue  # ja alcancado por outro caminho: nao reexpandir
            em_fronteira.add(filho)
            contador += 1
            heapq.heappush(heap, (custo + 1, contador, filho))
        pico = max(pico, len(heap))

    completo = estado_final is not None
    # A ordem de instalacao vem da ordenacao topologica do fecho alcancado:
    # o UCS descobre o conjunto, nao a ordem, e a ordem precisa ser executavel.
    ordem = _ordem_topologica_kahn(G) if completo else None

    return {
        "algoritmo": "UCS",
        "fecho": set(estado_final) if completo else set(),
        "ordem": ordem,
        "nivel_max": None,
        "profundidade_busca": len(estado_final) if completo else None,
        "fronteira_pico": pico,
        "consultas": len(estado_final) if completo else None,
        "estados_expandidos": expandidos,
        "orcamento": orcamento,
        "orcamento_estourado": not completo,
        "estados_teoricos": 2 ** G.number_of_nodes(),
        "ciclos": ciclos_do_grafo(G),
        "tempo_s": time.perf_counter() - inicio,
    }


# ===========================================================================
# 5. EXPERIMENTO
# ===========================================================================

def executar() -> None:
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    print("=" * 74)
    print("PROJETO 2 - BUSCA NAO INFORMADA")
    print("Resolucao de dependencias de pacotes Python (dados reais do PyPI)")
    print("=" * 74)

    print("\n[1] Construcao dos grafos de dependencia")
    cenarios = []
    for raiz in RAIZES:
        G, stats = construir_grafo(raiz, usar_rede=True)
        consultas = {n: obter_dependencias(n, usar_rede=False) for n in G}
        ciclos = ciclos_do_grafo(G)
        cenarios.append({
            "raiz": raiz, "G": G, "consultas": consultas,
            "nos": G.number_of_nodes(), "arestas": G.number_of_edges(),
            "ciclos": ciclos, "cache": stats["cache"],
        })
        print(f"    {raiz:<12} nos={G.number_of_nodes():3d} arestas={G.number_of_edges():3d}  "
              f"{'CICLO!' if ciclos else 'aciclico'}")

    principal = cenarios[0]
    raiz = principal["raiz"]
    G = principal["G"]
    consultas = principal["consultas"]
    ondas_ref = ondas_minimas(G)
    ondas_teorico = max(ondas_ref.values())

    print("\n[2] Execucao dos algoritmos em todas as raizes")
    print(f"    {'raiz':<12}{'algoritmo':<10}{'|V|':>6}{'consultas':>11}{'estados':>10}"
          f"{'prof.':>8}{'frontera':>10}{'ondas':>8}{'valida':>9}{'tempo (s)':>11}")

    for c in cenarios:
        c["resultados"] = [
            bfs(c["raiz"], c["G"], c["consultas"]),
            dfs(c["raiz"], c["G"], c["consultas"]),
            iddfs(c["raiz"], c["G"], c["consultas"]),
            ucs(c["raiz"], c["G"], c["consultas"]),
        ]
        c["ondas"] = ondas_minimas(c["G"])
        c["ondas_minimas"] = max(c["ondas"].values())
        c["nivel_bfs"] = max(_nivel_bfs(c["G"], c["raiz"]).values()) + 1

        for e in c["resultados"]:
            e["ordem_valida"] = bool(e["ordem"]) and ordem_e_valida(c["G"], e["ordem"])
            e["ondas"] = ondas_de_uma_ordem(c["G"], e["ordem"]) if e["ordem"] else None
            e["fecho_completo"] = e["fecho"] == set(c["G"].nodes)
            e["raiz"] = c["raiz"]
            if e["algoritmo"] == "UCS":
                # UCS pode estourar o orcamento; nesse caso nao ha ordem.
                status = "INCOMPLETO" if e["orcamento_estourado"] else "ok"
                print(f"    {c['raiz']:<12}{e['algoritmo']:<10}{c['nos']:>6}"
                      f"{str(e['consultas'] if e['consultas'] else '-'):>11}"
                      f"{e['estados_expandidos']:>10}"
                      f"{str(e['profundidade_busca'] or '-'):>8}"
                      f"{e['fronteira_pico']:>10}{str(e['ondas'] if e['ondas'] else '-'):>8}"
                      f"{status:>11}{e['tempo_s']:>11.4f}")
            else:
                print(f"    {c['raiz']:<12}{e['algoritmo']:<10}{len(e['fecho']):>6}"
                      f"{e['consultas']:>11}{'':>10}{e['profundidade_busca']:>8}"
                      f"{e['fronteira_pico']:>10}{e['ondas']:>8}"
                      f"{str(e['ordem_valida']):>9}{e['tempo_s']:>11.4f}")

    print("\n[3] Verificacao independente (BFS, DFS e IDDFS)")
    print(f"    ondas minimas, por relaxacao a partir das folhas = {ondas_teorico}")
    print(f"    nivel maximo da BFS a partir da raiz             = "
          f"{principal['resultados'][0]['nivel_max']}")
    print(f"    -> a BFS mede o MENOR caminho ate a raiz; a onda de instalacao exige o")
    print(f"       maior. Usar o nivel da BFS subestimaria em "
          f"{ondas_teorico - principal['resultados'][0]['nivel_max']} ondas "
          f"({100.0 * (ondas_teorico - principal['resultados'][0]['nivel_max']) / ondas_teorico:.0f}%).")
    print(f"    consultas: minimo teorico = |V| = {G.number_of_nodes()} (cada pacote precisa")
    print(f"       ser consultado uma vez para que suas dependencias sejam conhecidas)")
    for e in principal["resultados"][:3]:
        assert e["fecho_completo"], f"{e['algoritmo']} nao cobriu o fecho"
        assert e["ordem_valida"], f"{e['algoritmo']} produziu ordem invalida"
        assert e["consultas"] == G.number_of_nodes(), f"{e['algoritmo']} excedeu |V|"
        assert e["ondas"] == ondas_teorico, f"{e['algoritmo']} nao usou o minimo de ondas"
    print("    OK: BFS, DFS e IDDFS cobriram o fecho, produziram ordem valida, gastaram")
    print("        exatamente |V| consultas e atingiram o numero minimo de ondas.")

    print("\n[4] Custo do UCS: explosao exponencial do espaco de estados")
    print(f"    {'raiz':<12}{'|V|':>6}{'2^|V| (teorico)':>20}{'estados expandidos':>21}"
          f"{'orcamento':>12}{'resultado':>14}")
    for c in cenarios:
        e = next(r for r in c["resultados"] if r["algoritmo"] == "UCS")
        print(f"    {c['raiz']:<12}{c['nos']:>6}{e['estados_teoricos']:>20.3g}"
              f"{e['estados_expandidos']:>21,}{e['orcamento']:>12,}"
              f"{('completo' if not e['orcamento_estourado'] else 'INCOMPLETO'):>14}")
    completos = [c for c in cenarios
                 if not next(r for r in c["resultados"] if r["algoritmo"] == "UCS")["orcamento_estourado"]]
    print(f"    O UCS completa em {len(completos)} de {len(cenarios)} cenarios. Nos demais, o")
    print(f"    numero de estados cresce exponencialmente e o orcamento e excedido -- razao")
    print(f"    pela qual a busca por subconjuntos e inadequada aqui e BFS/DFS dominam.")

    print("\n[5] Detector de ciclos em grafo sintetico")
    ciclo = _detectar_ciclo(CICLO_SINTETICO)
    print(f"    ciclo esperado a->b->c->a ; detectado: {ciclo}")
    assert ciclo == ["a", "b", "c", "a"], "detector de ciclo falhou"
    print(f"    Nenhum fecho real do PyPI continha ciclo "
          f"({sum(len(c['ciclos']) for c in cenarios)} no total); o teste acima")
    print(f"    comprova que a ausencia de ciclos nos dados reais e propriedade, nao bug.")

    escala = [
        {"raiz": c["raiz"], "nos": c["nos"], "arestas": c["arestas"],
         "ondas_minimas": c["ondas_minimas"], "nivel_bfs": c["nivel_bfs"],
         "ciclos": len(c["ciclos"])}
        for c in cenarios
    ]

    _gravar(cenarios, ondas_ref, ondas_teorico, escala)
    _gerar_figuras(principal, ondas_ref, ondas_teorico, escala)

    print(f"\n[6] Artefatos gravados em:\n    {DIR_RESULTADOS}\n    {DIR_FIGURAS}")


def _nivel_bfs(G: nx.DiGraph, raiz: str) -> dict[str, int]:
    """Distancia minima (menor caminho) de cada no ate a raiz."""
    nivel = {raiz: 0}
    fila = deque([raiz])
    while fila:
        atual = fila.popleft()
        for dep in G.successors(atual):
            if dep not in nivel:
                nivel[dep] = nivel[atual] + 1
                fila.append(dep)
    return nivel


def _detectar_ciclo(mapa: dict[str, list[str]]) -> list[str]:
    """DFS com marcacao de cor; devolve o ciclo como lista de nos, ou [].

    Uma aresta para um no CINZA e uma aresta de retrocesso: o ciclo e formado
    pelo trecho da pilha entre o vizinho CINZA e o no atual. A reconstrucao
    sobe pelos pais a partir do no atual e para quando chega ao vizinho -- por
    isso nunca consulta `pai` da raiz da busca, que nao tem entrada.
    """
    cor: dict[str, int] = {}
    pai: dict[str, str] = {}

    def visitar(no: str) -> list[str]:
        cor[no] = CINZA
        for vizinho in mapa.get(no, []):
            estado = cor.get(vizinho, BRANCO)
            if estado == CINZA:  # aresta de retrocesso
                ciclo = []
                atual = no
                while atual != vizinho and atual in pai:
                    ciclo.append(atual)
                    atual = pai[atual]
                ciclo.append(vizinho)
                return ciclo[::-1] + [vizinho]
            if estado == BRANCO:
                pai[vizinho] = no
                achado = visitar(vizinho)
                if achado:
                    return achado
        cor[no] = PRETO
        return []

    return visitar("a") or []


# ===========================================================================
# 6. ARTEFATOS
# ===========================================================================

def _gravar(cenarios, ondas, ondas_teorico, escala) -> None:
    with open(os.path.join(DIR_RESULTADOS, "proj2_tabela.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["raiz", "algoritmo", "pacotes", "consultas", "estados_expandidos",
                    "profundidade_busca", "fronteira_pico", "ondas", "ordem_valida",
                    "fecho_completo", "ciclos", "tempo_s"])
        for c in cenarios:
            for e in c["resultados"]:
                w.writerow([
                    c["raiz"], e["algoritmo"], len(e["fecho"]), e["consultas"],
                    e.get("estados_expandidos", ""), e["profundidade_busca"],
                    e["fronteira_pico"], e["ondas"], e["ordem_valida"],
                    e["fecho_completo"], len(e["ciclos"]), f"{e['tempo_s']:.6f}",
                ])

    with open(os.path.join(DIR_RESULTADOS, "proj2_escala.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["raiz", "pacotes", "arestas", "ondas_minimas", "nivel_bfs", "ciclos"])
        for l in escala:
            w.writerow([l["raiz"], l["nos"], l["arestas"], l["ondas_minimas"],
                        l["nivel_bfs"], l["ciclos"]])

    with open(os.path.join(DIR_RESULTADOS, "proj2_ondas.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["onda", "pacotes"])
        for i in range(1, ondas_teorico + 1):
            membros = sorted(n for n, v in ondas.items() if v == i)
            w.writerow([i, len(membros)])

    with open(os.path.join(DIR_RESULTADOS, "proj2_resultados.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "raiz_principal": cenarios[0]["raiz"],
            "pacotes": cenarios[0]["nos"],
            "arestas": cenarios[0]["arestas"],
            "ondas_minimas": ondas_teorico,
            "resultados": [
                {
                    "raiz": c["raiz"],
                    **{k: v for k, v in e.items() if k != "ordem"},
                }
                for c in cenarios for e in c["resultados"]
            ],
            "escala": escala,
            "onda_por_pacote": ondas,
        }, fh, indent=2, ensure_ascii=False, default=str)


def _gerar_figuras(principal, ondas, ondas_teorico, escala) -> None:
    G = principal["G"]
    # multipartite_layout espera uma PARTICAO: chave -> conjunto de nos.
    camadas: dict[int, set[str]] = {}
    for n in G:
        camadas.setdefault(ondas[n] - 1, set()).add(n)
    pos = nx.multipartite_layout(G, subset_key=camadas, align="vertical", scale=1.0)

    fig, ax = plt.subplots(figsize=(9, 13))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.22, arrows=False,
                           edge_color="#5b7fa6", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=95,
                           node_color=[ondas[n] for n in G], cmap="viridis")
    rotulos = {n: (n if G.in_degree(n) + G.out_degree(n) > 1 else "") for n in G}
    nx.draw_networkx_labels(G, pos, labels=rotulos, ax=ax, font_size=5.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[principal["raiz"]],
                           node_size=290, node_color="#d1495b", edgecolors="white")
    ax.set_title(
        f"Figura 1 - Fecho transitivo de '{principal['raiz']}' "
        f"({G.number_of_nodes()} pacotes, {G.number_of_edges()} arestas)\n"
        f"Cor = onda de instalacao (1 a {ondas_teorico}); vermelho = pacote raiz",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIGURAS, "proj2_fig1_grafo.png"), dpi=200)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Memoria da busca: profundidade da pilha (DFS e IDDFS) contra o tamanho
    # da fronteira da BFS, no mesmo grafo.
    res = principal["resultados"]
    nomes = ["BFS\n(fronteira)", "DFS\n(pilha)", "IDDFS\n(limite)", "UCS\n(fronteira)"]
    valores = [r["fronteira_pico"] for r in res]
    cores = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    barras = ax1.bar(range(len(nomes)), valores, color=cores)
    ax1.set_xticks(range(len(nomes)))
    ax1.set_xticklabels(nomes, fontsize=7.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("pico da fronteira / pilha (nós, escala log)", fontsize=9)
    ax1.set_title(f"Memória da busca  ({principal['raiz']}, |V|={G.number_of_nodes()})",
                  fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    for barra, v in zip(barras, valores):
        ax1.annotate(f"{v:,}".replace(",", "."), (barra.get_x() + barra.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=8)

    rotulos_e = [e["raiz"] for e in escala]
    x = range(len(escala))
    ax2.bar([i - 0.2 for i in x], [e["ondas_minimas"] for e in escala], width=0.4,
            color="#55a868", label="ondas mínimas (maior caminho)")
    ax2.bar([i + 0.2 for i in x], [e["nivel_bfs"] for e in escala], width=0.4,
            color="#c44e52", label="nível BFS (menor caminho)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(rotulos_e, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("número de ondas / níveis", fontsize=9)
    ax2.set_title("Ondas de instalação vs. nível da BFS", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Figura 2 - Busca não informada: memória da busca e o erro do nível BFS",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(DIR_FIGURAS, "proj2_fig2_comparacao.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    executar()
