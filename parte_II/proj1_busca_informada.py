#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 1 - BUSCA INFORMADA
===========================
Roteamento de um drone de carga em area urbana real, evitando zonas de
interdicao de voo (aerodromos/helipontos e hospitais) extraidas do OpenStreetMap.

PROBLEMA FORMAL (Russell & Norvig, AIMA 4a ed., capitulo 3)
  - Agente      : drone de carga com deposito na UnB
  - Estados     : nos da malha viaria (arestas = corredores de voo)
  - Acoes       : voo de um no a um no adjacente
  - Custo c(n,m): tempo de voo em segundos (comprimento / velocidade de cruzeiro)
  - Objetivo    : menor custo do inicio ao fim sem sobrevoar zonas interditadas
  - Restricao   : "dura". As arestas que cruzam uma zona de interdicao sao
                  REMOVIDAS do grafo. Como o grafo resultante ja codifica a
                  restricao, nao ha necessidade de buscar estados que a violam.

ALGORITMOS IMPLEMENTADOS E COMPARADOS
  1. Dijkstra                  - referencia de custo otimo; tambem fornece h*
  2. A* (h geometrica)         - h(n) = distancia em linha reta / v_max
  3. A* (h* perfeita)          - h*(n) = d*(n, goal), via Dijkstra reverso
  4. Greedy Best-First         - ordena a fronteira apenas por h
  5. A* ponderado (W=1.5,2,3) - f = g + W*h, compromisso velocidade/qualidade

DEPENDENCIAS
  requests (download do OSM), numpy (geometria vetorizada),
  networkx (componentes conexas), matplotlib (figuras).
  Nao ha dependencia de Overpass/osmnx: usa-se a API 0.6 do OSM, mais estavel.

EXECUCAO
  python proj1_busca_informada.py
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import time
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle

# ===========================================================================
# 1. CONFIGURACAO DO EXPERIMENTO
# ===========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DADOS = os.path.join(BASE, "data")
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

# A API 0.6 do OSM exige identificacao do cliente.
USER_AGENT = "AI-Portfolio-FGA0221/1.0 (educational; g.alvesdearaujo06@gmail.com)"
URL_OSM = "https://api.openstreetmap.org/api/0.6/map?bbox={bbox}"

# AREA OPERACIONAL: Asa Norte / UnB ate o Autodromo Internacional de Brasilia
# (aprox. 4,8 km x 2,7 km). A area e dividida em faixas porque a API 0.6 do OSM
# limita cada requisicao a 50.000 nos. A API devolve as vias COMPLETAS, sem
# recorte; portanto a uniao das faixas, feita por identificador do OSM, nao
# cria arestas artificiais na fronteira.
FAIXAS = {
    "Oeste": (-47.9075, -15.7900, -47.8915, -15.7660),
    "Centro": (-47.8915, -15.7900, -47.8755, -15.7660),
    "Leste": (-47.8755, -15.7900, -47.8620, -15.7660),
}

# Hierarquia viaria usada como corredor de voo. Excluidos de proposito:
# footway, cycleway, path, steps, track e pedestrian, por nao serem
# corredores uteis a um drone de carga. "service" foi mantida por integrar a
# malha urbana conectando quadras.
HIERARQUIA_VIARIA = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "road", "service",
}

# Velocidade de cruzeiro por tipo de via (km/h), usada para converter
# comprimento em tempo de voo. O tag "maxspeed" do OSM tem precedencia.
VELOCIDADES = {
    "motorway": 80, "trunk": 65, "primary": 55, "secondary": 45,
    "tertiary": 40, "unclassified": 30, "residential": 30,
    "living_street": 15, "road": 30, "service": 20,
}
VELOCIDADE_PADRAO_KMH = 30.0
VELOCIDADE_MAXIMA_KMH = 80.0  # teto do modelo: base da heuristica admissivel

# Raios (m) das zonas de interdicao. Parametros DIDATICOS: representam a ordem
# de grandeza das restricoes regulatorias brasileiras. A ICA 100-40 classifica
# helipontos como aerodromos, e a ANAC exige autorizacao previa para operacao
# na proximidade de aerodromos registrados. Nao sao valores certificados -
# ver "Consideracoes Finais" do relatorio e a analise de sensibilidade.
RAIO_AERODROMO_M = 600.0
RAIO_HOSPITAL_M = 200.0

# Distancia (m) abaixo da qual duas zonas da mesma classe sao fundidas em uma.
# O OSM marca cada predio de hospital individualmente; tratar cada um como um
# circulo independente de 200 m fragmentaria a malha de forma artificial e sem
# significado operacional. A fusao agrega pontos em uma zona institucional.
FUSAO_ZONAS_M = 250.0

# Cenarios usados na analise de sensibilidade dos raios.
CENARIOS_SENSIBILIDADE = [
    ("permissivo", 400.0, 150.0),
    ("adotado", RAIO_AERODROMO_M, RAIO_HOSPITAL_M),
    ("restritivo", 900.0, 350.0),
]

# Pontos notaveis, usados apenas como anotacoes da figura.
REFERENCIAS = {
    "UnB (Darcy Ribeiro)": (-15.7730, -47.8700),
    "Autodromo Int. de Brasilia": (-15.7761, -47.9004),
}

# Cenario de origem e destino. Se o destino de referencia ficar inacessivel
# apos aplicar as interdicoes, ele e redefinido automaticamente para o vertice
# mais distante da origem dentro da mesma componente conexa.
ORIGEM_REF = (-15.7730, -47.8700)   # campus da UnB (deposito de carga)
DESTINO_REF = (-15.7761, -47.9004)  # Autodromo Internacional

# Amostragem da geometria das arestas para detectar cruzamento com as zonas.
PASSO_AMOSTRAGEM_M = 30.0
MAX_AMOSTRAS_POR_ARESTA = 12

RAIO_TERRA_M = 6_371_000.0


# ===========================================================================
# 2. AQUISICAO E CACHE DOS DADOS OPENSTREETMAP
# ===========================================================================

def obter_dados_osm(usar_rede: bool = True) -> tuple[dict, dict, list[ET.Element]]:
    """Baixa (ou le do cache) a area operacional.

    Retorna (nos, vias, arvores_xml):
      nos        : {id: {'id', 'lat', 'lon'}}
      vias       : {id: {'id', 'refs': [...], 'tags': {...}}} so para a
                   hierarquia viaria relevante
      arvores_xml: uma raiz <osm> por faixa, usada para extrair os POIs
                   (helipontos e hospitais quase sempre sao Ways, nao Nodes)
    """
    os.makedirs(DIR_DADOS, exist_ok=True)
    requests = None
    if usar_rede:
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Pacote 'requests' ausente. Rode a partir do cache (data/osm_*.osm) "
                "ou instale com: pip install requests"
            ) from exc

    nos: dict[str, dict] = {}
    vias: dict[str, dict] = {}
    arvores: list[ET.Element] = []

    for nome, (w, s, e, n) in FAIXAS.items():
        caminho = os.path.join(DIR_DADOS, f"osm_{nome}.osm")
        origem = None
        conteudo = None

        if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
            with open(caminho, "rb") as fh:
                conteudo = fh.read()
            origem = "cache"
        elif usar_rede:
            bbox = f"{w},{s},{e},{n}"
            resposta = requests.get(
                URL_OSM.format(bbox=bbox), headers={"User-Agent": USER_AGENT}, timeout=200
            )
            if resposta.status_code != 200:
                raise SystemExit(f"Falha ao baixar a faixa {nome}: HTTP {resposta.status_code}")
            conteudo = resposta.content
            with open(caminho, "wb") as fh:
                fh.write(conteudo)
            origem = "rede"
            time.sleep(1.0)  # intervalo educado entre requisicoes
        else:
            continue

        raiz = ET.fromstring(conteudo)
        arvores.append(raiz)
        for elemento in raiz.findall("node"):
            nos[elemento.get("id")] = {
                "id": elemento.get("id"),
                "lat": float(elemento.get("lat")),
                "lon": float(elemento.get("lon")),
            }
        for elemento in raiz.findall("way"):
            tags = {t.get("k"): t.get("v") for t in elemento.findall("tag")}
            if tags.get("highway") not in HIERARQUIA_VIARIA:
                continue
            vias[elemento.get("id")] = {
                "id": elemento.get("id"),
                "refs": [nd.get("ref") for nd in elemento.findall("nd")],
                "tags": tags,
            }
        print(f"    faixa {nome:6s} [{origem:5s}] "
              f"{len(raiz.findall('node')):6d} nos, {len(raiz.findall('way')):5d} vias")

    if not vias:
        raise SystemExit(
            "Nenhuma via obtida. Execute uma vez com rede ou forneca data/osm_*.osm"
        )
    return nos, vias, arvores


# ===========================================================================
# 3. GRAFO DE VOO: COORDENADAS E CONSTRUCAO
# ===========================================================================

def _para_metros(lats: np.ndarray, lons: np.ndarray, lat0: float, lon0: float):
    """Projeta (lat, lon) para um plano local em metros.

    Projecao equirretangular centrada na area. Numa extensao de ~5 km o erro de
    escala e inferior a 0,01%, muito abaixo do ruido geometrico do proprio dado
    do OSM, e dispensa bibliotecas de projecao geodesica.
    """
    k = math.cos(math.radians(lat0))
    x = RAIO_TERRA_M * np.radians(lons - lon0) * k
    y = RAIO_TERRA_M * np.radians(lats - lat0)
    return x, y


def _para_xy(lat: float, lon: float, lat0: float, lon0: float, k: float):
    return (RAIO_TERRA_M * math.radians(lon - lon0) * k,
            RAIO_TERRA_M * math.radians(lat - lat0))


def _velocidade(tags: dict) -> float:
    """Velocidade de cruzeiro em m/s, a partir de maxspeed ou da hierarquia."""
    bruto = tags.get("maxspeed")
    if bruto:
        digitos = "".join(c for c in str(bruto) if c.isdigit())
        if digitos:
            return float(digitos) / 3.6
    hier = tags.get("highway", "")
    return VELOCIDADES.get(hier.replace("_link", ""), VELOCIDADE_PADRAO_KMH) / 3.6


def construir_grafo(nos: dict, vias: dict) -> tuple[nx.Graph, dict]:
    """Monta o grafo de voo, ponderado por tempo de voo (segundos).

    O grafo e INDIRETO e as arestas nao respeitam oneway: o drone nao esta
    obrigado a seguir o sentido da via, e impor essa restricao multiplicaria
    por ~2 o numero de estados sem justificar fisicamente para a tarefa.
    """
    lat0 = float(np.mean([n["lat"] for n in nos.values()]))
    lon0 = float(np.mean([n["lon"] for n in nos.values()]))
    k = math.cos(math.radians(lat0))

    G = nx.Graph()
    for no in nos.values():
        x, y = _para_xy(no["lat"], no["lon"], lat0, lon0, k)
        G.add_node(no["id"], x=x, y=y, lat=no["lat"], lon=no["lon"])

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    for via in vias.values():
        v_ms = _velocidade(via["tags"])
        refs = [r for r in via["refs"] if r in pos]
        for a, b in zip(refs, refs[1:]):
            if a == b:
                continue
            xa, ya = pos[a]
            xb, yb = pos[b]
            comprimento = math.hypot(xb - xa, yb - ya)
            if comprimento <= 0.0:
                continue
            custo = comprimento / v_ms
            # Em caso de multiaresta entre o mesmo par, mantem a mais rapida.
            if G.has_edge(a, b) and G[a][b]["custo"] <= custo:
                continue
            G.add_edge(a, b, custo=custo, comprimento=comprimento, velocidade=v_ms)

    G.remove_nodes_from([n for n in G.nodes if G.degree(n) == 0])
    proj = {"lat0": lat0, "lon0": lon0, "k": k}
    return G, proj


# ===========================================================================
# 4. ZONAS DE INTERDICAO E APLICACAO DAS RESTRICOES
# ===========================================================================

def _classificar(tags: dict) -> str | None:
    """Classifica um elemento OSM como zona de interdicao, ou None.

    A ICA 100-40 classifica helipontos como aerodromos, por isso
    aeroway=helipad entra na classe AERODROMO.
    """
    if tags.get("aeroway") in {"aerodrome", "helipad", "airstrip"}:
        return "AERODROMO"
    if tags.get("amenity") in {"hospital", "clinic"}:
        return "HOSPITAL"
    if tags.get("healthcare") in {"hospital", "clinic"}:
        return "HOSPITAL"
    if tags.get("building") == "hospital":
        return "HOSPITAL"
    return None


def extrair_zonas(nos: dict, arvores: list[ET.Element], proj: dict) -> list[dict]:
    """Extrai centros e raios das zonas de interdicao do proprio download OSM."""
    lat0, lon0, k = proj["lat0"], proj["lon0"], proj["k"]
    zonas: list[dict] = []
    vistas: set[str] = set()

    def registrar(chave, classe, nome, xy):
        if chave in vistas:
            return
        vistas.add(chave)
        zonas.append({
            "id": chave,
            "classe": classe,
            "nome": nome or f"{classe} {chave}",
            "x": float(xy[0]),
            "y": float(xy[1]),
            "raio": RAIO_AERODROMO_M if classe == "AERODROMO" else RAIO_HOSPITAL_M,
        })

    for raiz in arvores:
        # --- Nos ---
        for elemento in raiz.findall("node"):
            tags = {t.get("k"): t.get("v") for t in elemento.findall("tag")}
            classe = _classificar(tags)
            if not classe:
                continue
            lat, lon = float(elemento.get("lat")), float(elemento.get("lon"))
            registrar(f"n{elemento.get('id')}", classe, tags.get("name"),
                      _para_xy(lat, lon, lat0, lon0, k))

        # --- Ways ---
        for elemento in raiz.findall("way"):
            tags = {t.get("k"): t.get("v") for t in elemento.findall("tag")}
            classe = _classificar(tags)
            if not classe:
                continue
            refs = [nd.get("ref") for nd in elemento.findall("nd")]
            pts = [_para_xy(nos[r]["lat"], nos[r]["lon"], lat0, lon0, k)
                   for r in refs if r in nos]
            if not pts:
                continue
            if len(refs) > 2 and refs[0] == refs[-1]:
                # Via fechada (poligono do terreno): centride dos vertices.
                centro = (sum(p[0] for p in pts) / len(pts),
                          sum(p[1] for p in pts) / len(pts))
            else:
                # Via aberta (pista, heliponto): ponto medio do segmento central.
                centro = pts[len(pts) // 2]
            registrar(f"w{elemento.get('id')}", classe, tags.get("name"), centro)

    return zonas


def agregar_zonas(zonas: list[dict], fusao: float) -> list[dict]:
    """Funde zonas da mesma classe cujos centros distam menos que `fusao`.

    Uma unidade hospitalar e um heliponto sao, na pratica, um unico ponto de
    interdicao; o OSM apenas os representa por varios elementos (um por predio).
    Sem esta agregacao, cada hospital viraria um circulo independente e a malha
    seria fragmentada de forma artificial, sem correspondencia com a operacao
    real de um drone.

    O agrupamento e guloso e dependente da ordem de entrada; por isso a entrada
    e ordenada por (classe, id) para tornar o resultado deterministico.
    """
    grupos: list[dict] = []
    for z in sorted(zonas, key=lambda z: (z["classe"], z["id"])):
        for g in grupos:
            if g["classe"] != z["classe"]:
                continue
            if math.dist((g["x"], g["y"]), (z["x"], z["y"])) <= fusao:
                n = g["_n"]
                g["x"] = (g["x"] * n + z["x"]) / (n + 1)
                g["y"] = (g["y"] * n + z["y"]) / (n + 1)
                g["_n"] = n + 1
                g["raio"] = max(g["raio"], z["raio"])
                g["membros"].append(z["nome"])
                break
        else:
            grupos.append({
                "classe": z["classe"], "x": z["x"], "y": z["y"],
                "raio": z["raio"], "_n": 1, "membros": [z["nome"]],
            })

    for g in grupos:
        membros = g.pop("membros")
        g["nome"] = membros[0] if len(membros) == 1 else f"Agrupamento de {len(membros)} pontos"
    return grupos


def _zona_mais_perto(zonas: list[dict], x: float, y: float):
    if not zonas:
        return None
    return min(zonas, key=lambda z: math.dist((z["x"], z["y"]), (x, y)))


def aplicar_restricoes(G: nx.Graph, zonas: list[dict]) -> tuple[int, int]:
    """Remove as arestas que sobrevoam qualquer zona de interdicao.

    A geometria de cada aresta e amostrada com passo constante; se qualquer
    amostra cair dentro de um cilindro de interdicao, a aresta e removida.
    Retorna (numero removido, numero total antes da remocao).
    """
    arestas = list(G.edges(data=True))
    amostras: list[tuple[float, float]] = []
    dono: list[int] = []

    for i, (u, v, dados) in enumerate(arestas):
        xa, ya = G.nodes[u]["x"], G.nodes[u]["y"]
        xb, yb = G.nodes[v]["x"], G.nodes[v]["y"]
        comp = dados["comprimento"]
        k = int(min(MAX_AMOSTRAS_POR_ARESTA, max(2, math.ceil(comp / PASSO_AMOSTRAGEM_M))))
        for t in np.linspace(0.0, 1.0, k):
            amostras.append((xa + t * (xb - xa), ya + t * (yb - ya)))
            dono.append(i)

    P = np.asarray(amostras)
    dono = np.asarray(dono, dtype=np.int64)
    interditadas = np.zeros(P.shape[0], dtype=bool)
    for z in zonas:
        d = np.hypot(P[:, 0] - z["x"], P[:, 1] - z["y"])
        interditadas |= d <= z["raio"]

    for i in sorted({int(j) for j in dono[interditadas]}):
        u, v, _ = arestas[i]
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    return len(sorted(set(dono[interditadas].tolist()))), len(arestas)


# ===========================================================================
# 5. ALGORITMOS DE BUSCA
# ===========================================================================
# Todos compartilham a mesma assinatura e devolvem o mesmo dicionario de
# metricas, o que garante uma comparacao justa entre eles.

def _reconstruir(parente, destino):
    if destino not in parente:
        return None
    caminho = [destino]
    while parente[caminho[-1]] is not None:
        caminho.append(parente[caminho[-1]])
    return caminho[::-1]


def _montar_resultado(nome, caminho, custo, expandidos, fronteira_max, tempo, extra=None):
    res = {
        "algoritmo": nome,
        "encontrou": caminho is not None,
        "custo_s": None if custo is None else float(custo),
        "nos_expandidos": int(expandidos),
        "fronteira_max": int(fronteira_max),
        "tempo_s": float(tempo),
        "profundidade": None if caminho is None else len(caminho) - 1,
        "caminho": caminho,
    }
    if extra:
        res.update(extra)
    return res


def busca_dijkstra(G, origem, destino):
    """Busca de custo uniforme. Fornece o custo otimo de referencia e, quando
    executada do destino para a origem, a heuristica perfeita h*."""
    inicio = time.perf_counter()
    g = {origem: 0.0}
    parente = {origem: None}
    F = [(0.0, origem)]
    expandidos = 0
    fronteira_max = 0
    while F:
        custo_atual, u = heapq.heappop(F)
        if custo_atual > g.get(u, math.inf) + 1e-12:
            continue
        if u == destino:
            break
        expandidos += 1
        for v, dados in G[u].items():
            novo = custo_atual + dados["custo"]
            if novo < g.get(v, math.inf) - 1e-12:
                g[v] = novo
                parente[v] = u
                heapq.heappush(F, (novo, v))
        fronteira_max = max(fronteira_max, len(F))
    return _montar_resultado(
        "Dijkstra", _reconstruir(parente, destino), g.get(destino),
        expandidos, fronteira_max, time.perf_counter() - inicio, {"g_map": g},
    )


def busca_astar(G, origem, destino, h, peso: float = 1.0, nome: str | None = None):
    """A* com f(n) = g(n) + peso * h(n).

    peso = 1  -> A* classico; h admissivel garante solucao otima.
    peso > 1  -> A* ponderado; h deixa de ser admissivel, a busca fica mais
                 rapida e a solucao possivelmente subotima.
    """
    inicio = time.perf_counter()
    g = {origem: 0.0}
    parente = {origem: None}
    fechado: set = set()
    F = [(peso * h(origem), 0.0, origem)]
    expandidos = 0
    fronteira_max = 0
    while F:
        _, custo_atual, u = heapq.heappop(F)
        if u in fechado:
            continue
        if u == destino:
            break
        fechado.add(u)
        expandidos += 1
        for v, dados in G[u].items():
            if v in fechado:
                continue
            novo = custo_atual + dados["custo"]
            if novo < g.get(v, math.inf) - 1e-12:
                g[v] = novo
                parente[v] = u
                heapq.heappush(F, (novo + peso * h(v), novo, v))
        fronteira_max = max(fronteira_max, len(F))
    return _montar_resultado(
        nome or f"A* (peso={peso:g})", _reconstruir(parente, destino), g.get(destino),
        expandidos, fronteira_max, time.perf_counter() - inicio,
    )


def busca_greedy(G, origem, destino, h):
    """Greedy Best-First: ordena a fronteira apenas por h(n), ignorando g(n)."""
    inicio = time.perf_counter()
    g = {origem: 0.0}
    parente = {origem: None}
    visto: set = set()
    F = [(h(origem), origem)]
    expandidos = 0
    fronteira_max = 0
    while F:
        _, u = heapq.heappop(F)
        if u in visto:
            continue
        if u == destino:
            break
        visto.add(u)
        expandidos += 1
        for v, dados in G[u].items():
            if v in visto:
                continue
            if v not in parente:
                parente[v] = u
                g[v] = g[u] + dados["custo"]
            heapq.heappush(F, (h(v), v))
        fronteira_max = max(fronteira_max, len(F))
    return _montar_resultado(
        "Greedy Best-First", _reconstruir(parente, destino), g.get(destino),
        expandidos, fronteira_max, time.perf_counter() - inicio,
    )


def medir_qualidade_heuristica(G, origem, h, h_perfeita) -> dict:
    """Estatisticas de h(n)/h*(n) sobre os nos efetivamente visitados por A*."""
    amostras = []
    g = {origem: 0.0}
    F = [(0.0, origem)]
    while F:
        custo_atual, u = heapq.heappop(F)
        if custo_atual > g.get(u, math.inf) + 1e-12:
            continue
        hv, hp = h(u), h_perfeita.get(u)
        if hp is not None and hp > 1e-9:
            amostras.append(hv / hp)
        for v, dados in G[u].items():
            novo = custo_atual + dados["custo"]
            if novo < g.get(v, math.inf) - 1e-12:
                g[v] = novo
                heapq.heappush(F, (novo, v))
    a = np.asarray(amostras)
    return {
        "amostras": int(a.size),
        "razao_mediana": float(np.median(a)) if a.size else None,
        "razao_media": float(a.mean()) if a.size else None,
        "razao_p90": float(np.percentile(a, 90)) if a.size else None,
        "razao_min": float(a.min()) if a.size else None,
    }


# ===========================================================================
# 6. EXPERIMENTO
# ===========================================================================

def _no_mais_proximo(G, lat, lon, proj):
    x = RAIO_TERRA_M * math.radians(lon - proj["lon0"]) * proj["k"]
    y = RAIO_TERRA_M * math.radians(lat - proj["lat0"])
    return min(G.nodes, key=lambda n: (G.nodes[n]["x"] - x) ** 2 + (G.nodes[n]["y"] - y) ** 2)


def com_raios(zonas: list[dict], raio_aero: float, raio_hosp: float) -> list[dict]:
    """Devolve copia das zonas com os raios do cenario considerado."""
    return [
        {**z, "raio": raio_aero if z["classe"] == "AERODROMO" else raio_hosp}
        for z in zonas
    ]


def _h_euclidiana(G: nx.Graph, destino):
    """Heuristica admissivel: distancia em linha reta dividida por v_max."""
    v_max = VELOCIDADE_MAXIMA_KMH / 3.6
    gx, gy = G.nodes[destino]["x"], G.nodes[destino]["y"]

    def h(no):
        # Nenhuma rota e mais curta que a reta, e nenhuma e mais rapida que
        # v_max; logo h(n) <= custo real de qualquer caminho de n ate o goal.
        return math.hypot(G.nodes[no]["x"] - gx, G.nodes[no]["y"] - gy) / v_max

    return h


def _vertice_mais_distante(G: nx.Graph, origem):
    return max(
        G.nodes,
        key=lambda n: (G.nodes[n]["x"] - G.nodes[origem]["x"]) ** 2
        + (G.nodes[n]["y"] - G.nodes[origem]["y"]) ** 2,
    )


def zonas_atravessadas(G: nx.Graph, caminho, zonas: list[dict]) -> list[dict]:
    """Zonas de interdicao cobertas por algum vertice do caminho."""
    if not caminho:
        return []
    atingidas = []
    for z in zonas:
        for no in caminho:
            d = math.hypot(G.nodes[no]["x"] - z["x"], G.nodes[no]["y"] - z["y"])
            if d <= z["raio"]:
                atingidas.append(z)
                break
    return atingidas


def _cenario_sensibilidade(G_livre, zonas_base, origem, nome, raio_aero, raio_hosp):
    """Repete o experimento com outros raios, mantendo origem e grafo base."""
    G = G_livre.copy()
    zonas = com_raios(zonas_base, raio_aero, raio_hosp)
    n_rem, n_tot = aplicar_restricoes(G, zonas)
    componentes = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(componentes[0]).copy()

    base = {
        "cenario": nome,
        "raio_aerodromo_m": raio_aero,
        "raio_hospital_m": raio_hosp,
        "arestas_removidas_pct": 100.0 * n_rem / max(n_tot, 1),
        "arestas_removidas": n_rem,
        "nos": G.number_of_nodes(),
        "arestas": G.number_of_edges(),
        "componentes": len(componentes),
    }
    if origem not in G:
        return {**base, "viavel": False}

    destino = _vertice_mais_distante(G, origem)
    dij = busca_dijkstra(G, origem, destino)
    astar = busca_astar(G, origem, destino, _h_euclidiana(G, destino), 1.0, "A* (h geometrica)")
    dij.pop("g_map")
    d_reta = math.dist(
        (G.nodes[origem]["x"], G.nodes[origem]["y"]),
        (G.nodes[destino]["x"], G.nodes[destino]["y"]),
    )
    return {
        **base,
        "viavel": True,
        "distancia_reta_m": d_reta,
        "custo_otimo_s": dij["custo_s"],
        "dijkstra_expandidos": dij["nos_expandidos"],
        "astar_expandidos": astar["nos_expandidos"],
        "reducao_expandidos_pct": 100.0 * (dij["nos_expandidos"] - astar["nos_expandidos"])
        / max(dij["nos_expandidos"], 1),
    }


def executar():
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    print("=" * 74)
    print("PROJETO 1 - BUSCA INFORMADA")
    print("Roteamento de drone com zonas de interdicao sobre dados OSM")
    print("=" * 74)

    # --- Etapa 1: dados --------------------------------------------------
    print("\n[1] Aquisicao dos dados OpenStreetMap")
    nos, vias, arvores = obter_dados_osm(usar_rede=True)
    print(f"    total: {len(nos)} nos, {len(vias)} vias da hierarquia viaria")

    # --- Etapa 2: grafo --------------------------------------------------
    print("\n[2] Construcao do grafo de voo")
    G, proj = construir_grafo(nos, vias)
    componentes = sorted(nx.connected_components(G), key=len, reverse=True)
    G_livre = G.subgraph(componentes[0]).copy()
    G = G_livre
    print(f"    nos={G.number_of_nodes()}  arestas={G.number_of_edges()}  "
          f"componentes={len(componentes)}  maior={len(componentes[0])}")

    # --- Etapa 3: zonas e restricoes -------------------------------------
    print("\n[3] Zonas de interdicao e aplicacao das restricoes")
    zonas_brutas = extrair_zonas(nos, arvores, proj)
    # Agrega por proximidade ANTES de aplicar raios: sem isso cada predio
    # hospitalar viraria um circulo isolado e a malha seria fragmentada de
    # forma artificial.
    zonas_agregadas = agregar_zonas(zonas_brutas, FUSAO_ZONAS_M)
    zonas = com_raios(zonas_agregadas, RAIO_AERODROMO_M, RAIO_HOSPITAL_M)
    n_aero = sum(1 for z in zonas if z["classe"] == "AERODROMO")
    print(f"    {len(zonas_brutas)} pontos OSM -> {len(zonas)} zonas "
          f"({n_aero} aerodromos/helipontos, {len(zonas) - n_aero} hospitals/clinicas)")
    for z in zonas:
        if z["classe"] == "AERODROMO":
            print(f"      AERODROMO  {z['nome']}  (r = {z['raio']:.0f} m)")

    G = G_livre.copy()
    n_rem, n_total_arestas = aplicar_restricoes(G, zonas)
    componentes = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(componentes[0]).copy()
    print(f"    arestas interditadas removidas: {n_rem} de {n_total_arestas} "
          f"({100.0 * n_rem / max(n_total_arestas, 1):.1f}%)")
    print(f"    grafo admissivel: {G.number_of_nodes()} nos, {G.number_of_edges()} arestas "
          f"({len(componentes)} componentes; maior={len(componentes[0])})")

    # --- Etapa 4: origem e destino ---------------------------------------
    print("\n[4] Definicao do cenario de origem e destino")
    origem = _no_mais_proximo(G, *ORIGEM_REF, proj)
    # Destino = vertice mais distante da origem em linha reta. Escolha
    # deterministica que maximiza a dificuldade da busca e percorre a area
    # operacional completa; evita que o destino caia por acaso ao lado da
    # origem etorne o experimento trivial.
    destino = _vertice_mais_distante(G, origem)
    d_reta = math.dist(
        (G.nodes[origem]["x"], G.nodes[origem]["y"]),
        (G.nodes[destino]["x"], G.nodes[destino]["y"]),
    )
    print(f"    origem (deposito UnB)  = {origem}")
    print(f"    destino (vertice mais distante) = {destino}")
    print(f"    distancia em linha reta = {d_reta:.0f} m")

    # --- Etapa 5: baseline sem restricoes --------------------------------
    print("\n[5] Baseline: rota otima SEM restricoes (mesmos extremos)")
    livre = busca_dijkstra(G_livre, origem, destino)
    violadas = zonas_atravessadas(G_livre, livre["caminho"], zonas)
    print(f"    custo={livre['custo_s']:.1f} s  nos expandidos={livre['nos_expandidos']}")
    print(f"    a rota livre sobrevoa {len(violadas)} zona(s) de interdicao:")
    for z in violadas:
        print(f"      - {z['classe']}: {z['nome']}")

    # --- Etapa 6: heuristicas --------------------------------------------
    print("\n[6] Heuristicas")
    h_euclidiana = _h_euclidiana(G, destino)
    reverso = busca_dijkstra(G, destino, origem)
    h_perfeita = reverso["g_map"]
    print(f"    h* perfeita alcancavel em {len(h_perfeita)} nos "
          f"(custo reverso={reverso['custo_s']:.1f} s)")

    # --- Etapa 7: experimentos -------------------------------------------
    print("\n[7] Execucao dos algoritmos no grafo admissivel")
    dijkstra = busca_dijkstra(G, origem, destino)
    dijkstra.pop("g_map")
    custo_otimo = dijkstra["custo_s"]
    print(f"    custo otimo={custo_otimo:.1f} s  vs {livre['custo_s']:.1f} s sem restricoes "
          f"({100.0 * (custo_otimo - livre['custo_s']) / livre['custo_s']:+.1f}%)")

    experimentos = [dijkstra]
    experimentos.append(busca_astar(G, origem, destino, h_euclidiana, 1.0, "A* (h geometrica)"))
    experimentos.append(
        busca_astar(G, origem, destino, lambda n: h_perfeita.get(n, math.inf), 1.0, "A* (h* perfeita)")
    )
    experimentos.append(busca_greedy(G, origem, destino, h_euclidiana))
    for peso in (1.5, 2.0, 3.0):
        experimentos.append(
            busca_astar(G, origem, destino, h_euclidiana, peso, f"A* ponderado (W={peso:g})")
        )

    print(f"\n    {'algoritmo':<24}{'custo (s)':>11}{'exp.:':>9}"
          f"{'fronteira':>11}{'gap (%)':>9}{'tempo (s)':>11}")
    for e in experimentos:
        e["gap_sobre_otimo_pct"] = (
            100.0 * (e["custo_s"] - custo_otimo) / custo_otimo if e["custo_s"] else None
        )
        e["comprimento_m"] = (
            sum(G[u][v]["comprimento"] for u, v in zip(e["caminho"], e["caminho"][1:]))
            if e["caminho"] else None
        )
        e["nos_expandidos_relativo"] = (
            e["nos_expandidos"] / dijkstra["nos_expandidos"]
            if dijkstra["nos_expandidos"] else None
        )
        gap = "n/d" if e["gap_sobre_otimo_pct"] is None else f"{e['gap_sobre_otimo_pct']:.2f}"
        print(f"    {e['algoritmo']:<24}{e['custo_s']:>11.1f}{e['nos_expandidos']:>9}"
              f"{e['fronteira_max']:>11}{gap:>9}{e['tempo_s']:>11.4f}")

    qualidade = medir_qualidade_heuristica(G, origem, h_euclidiana, h_perfeita)
    print(f"\n[8] Qualidade da heuristica geometrica h/h*: mediana="
          f"{qualidade['razao_mediana']:.3f}  p90={qualidade['razao_p90']:.3f}  "
          f"min={qualidade['razao_min']:.3f}  n={qualidade['amostras']}")

    # --- Etapa 9: sensibilidade ------------------------------------------
    print("\n[9] Sensibilidade aos raios das zonas de interdicao")
    sens = []
    for nome, ra_a, ra_h in CENARIOS_SENSIBILIDADE:
        c = _cenario_sensibilidade(G_livre, zonas_agregadas, origem, nome, ra_a, ra_h)
        sens.append(c)
        if not c["viavel"]:
            print(f"    {nome:<12} r={ra_a:.0f}/{ra_h:.0f} m  "
                  f"removidas={c['arestas_removidas_pct']:5.1f}%  INVIANVEL")
        else:
            print(f"    {nome:<12} r={ra_a:.0f}/{ra_h:.0f} m  "
                  f"removidas={c['arestas_removidas_pct']:5.1f}%  "
                  f"nos={c['nos']:5d}  custo={c['custo_otimo_s']:7.1f} s  "
                  f"reta={c['distancia_reta_m']:5.0f} m  "
                  f"Dijkstra={c['dijkstra_expandidos']:5d}  A*={c['astar_expandidos']:5d}")

    # --- Etapa 10: artefatos ---------------------------------------------
    caminho_a = next(e["caminho"] for e in experimentos if e["algoritmo"] == "A* (h geometrica)")

    resumo = {
        "cena": {
            "area": "Asa Norte / UnB - Autodromo Internacional de Brasilia, Brasilia-DF",
            "origem_referencia": list(ORIGEM_REF),
            "destino_referencia": list(DESTINO_REF),
            "origem": origem,
            "destino": destino,
            "distancia_reta_m": d_reta,
            "nos_osm": len(nos),
            "vias_osm": len(vias),
        },
        "grafo": {
            "nos": G.number_of_nodes(),
            "arestas": G.number_of_edges(),
            "arestas_removidas": n_rem,
            "arestas_antes": n_total_arestas,
            "zonas_brutas": len(zonas_brutas),
            "zonas": zonas,
        },
        "baseline_sem_restricoes": {
            "custo_s": livre["custo_s"],
            "nos_expandidos": livre["nos_expandidos"],
            "zonas_violadas": [
                {"classe": z["classe"], "nome": z["nome"], "raio_m": z["raio"]} for z in violadas
            ],
            "aumento_de_custo_pct": 100.0 * (custo_otimo - livre["custo_s"]) / livre["custo_s"],
            "caminho_diferente": list(livre["caminho"]) != list(caminho_a),
        },
        "custo_otimo_s": custo_otimo,
        "resultados": [{k: v for k, v in e.items() if k != "caminho"} for e in experimentos],
        "rota_a_star": caminho_a,
        "qualidade_heuristica": qualidade,
        "sensibilidade": sens,
        "parametros": {
            "raio_aerodromo_m": RAIO_AERODROMO_M,
            "raio_hospital_m": RAIO_HOSPITAL_M,
            "fusao_zonas_m": FUSAO_ZONAS_M,
            "velocidade_maxima_kmh": VELOCIDADE_MAXIMA_KMH,
            "passo_amostragem_m": PASSO_AMOSTRAGEM_M,
        },
    }
    with open(os.path.join(DIR_RESULTADOS, "proj1_resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, indent=2, ensure_ascii=False, default=float)

    with open(os.path.join(DIR_RESULTADOS, "proj1_tabela.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["algoritmo", "custo_s", "comprimento_m", "profundidade",
                    "nos_expandidos", "nos_expandidos_relativo", "fronteira_max",
                    "tempo_s", "gap_sobre_otimo_pct"])
        for e in experimentos:
            w.writerow([
                e["algoritmo"], f"{e['custo_s']:.3f}", f"{e['comprimento_m']:.1f}",
                e["profundidade"], e["nos_expandidos"],
                f"{e['nos_expandidos_relativo']:.4f}", e["fronteira_max"],
                f"{e['tempo_s']:.5f}", f"{e['gap_sobre_otimo_pct']:.4f}",
            ])

    with open(os.path.join(DIR_RESULTADOS, "proj1_sensibilidade.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cenario", "raio_aerodromo_m", "raio_hospital_m", "viavel",
                    "arestas_removidas_pct", "nos", "arestas", "distancia_reta_m",
                    "custo_otimo_s", "dijkstra_expandidos", "astar_expandidos",
                    "reducao_expandidos_pct"])
        for c in sens:
            w.writerow([
                c["cenario"], f"{c['raio_aerodromo_m']:.0f}", f"{c['raio_hospital_m']:.0f}",
                c["viavel"], f"{c['arestas_removidas_pct']:.2f}", c["nos"], c["arestas"],
                f"{c.get('distancia_reta_m', float('nan')):.1f}",
                f"{c.get('custo_otimo_s', float('nan')):.2f}",
                c.get("dijkstra_expandidos", ""), c.get("astar_expandidos", ""),
                f"{c.get('reducao_expandidos_pct', float('nan')):.2f}",
            ])

    _gerar_figuras(G, proj, zonas, experimentos, caminho_a, origem, destino)
    print(f"\n[10] Artefatos gravados em:\n    {DIR_RESULTADOS}\n    {DIR_FIGURAS}")


# ===========================================================================
# 7. FIGURAS
# ===========================================================================

def _gerar_figuras(G, proj, zonas, experimentos, caminho, origem, destino):
    # ---------- Figura 1: mapa ----------
    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=200)
    E = np.array([[G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[v]["x"], G.nodes[v]["y"]]
                  for u, v in G.edges()])
    ax.plot(E[:, 0], E[:, 1], lw=0.35, color="#c3ccd4", zorder=1)

    for z in zonas:
        cor = "#c0392b" if z["classe"] == "AERODROMO" else "#2e86c1"
        ax.add_patch(Circle((z["x"], z["y"]), z["raio"], facecolor=cor, alpha=0.12,
                            edgecolor=cor, lw=0.7, zorder=2))

    nome_aero = next((z["nome"] for z in zonas
                      if z["classe"] == "AERODROMO" and "Autódromo" in z["nome"]), None)
    if nome_aero:
        alvo = next(z for z in zonas if z["nome"] == nome_aero)
        ax.annotate(nome_aero, (alvo["x"], alvo["y"] - alvo["raio"] * 1.05),
                    fontsize=6.2, ha="center", va="top", color="#7b241c", zorder=6)

    for nome, (la, lo) in REFERENCIAS.items():
        px, py = _para_xy(la, lo, proj["lat0"], proj["lon0"], proj["k"])
        ax.scatter(px, py, marker="^", s=60, facecolor="none",
                   edgecolor="#333333", lw=1.1, zorder=6)
        ax.annotate(nome, (px, py), fontsize=6.2, color="#333333",
                    xytext=(0, -11), textcoords="offset points", ha="center", zorder=6)

    if caminho and len(caminho) > 1:
        xs = [G.nodes[n]["x"] for n in caminho]
        ys = [G.nodes[n]["y"] for n in caminho]
        ax.plot(xs, ys, lw=2.3, color="#1a7f37", zorder=4,
                label=f"A*: rota ótima ({len(caminho) - 1} trechos)")
        ax.scatter(xs, ys, s=3.5, color="#1a7f37", zorder=5)

    ax.scatter([G.nodes[origem]["x"]], [G.nodes[origem]["y"]], s=95, marker="o",
               color="#1a7f37", edgecolor="white", lw=1.1, zorder=7, label="Origem (UnB)")
    ax.scatter([G.nodes[destino]["x"]], [G.nodes[destino]["y"]], s=150, marker="*",
               color="#f39c12", edgecolor="white", lw=1.0, zorder=7, label="Destino")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m) - projecao local", fontsize=7.5)
    ax.set_ylabel("y (m)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.93)
    ax.set_title(
        "Figura 1 - Rota ótima de drone (A*) sobre a malha viária real do OSM.\n"
        "Vermelho: interdição de aeródromos/helipontos (900 m). Azul: hospitais (400 m).",
        fontsize=8.6,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIGURAS, "proj1_fig1_mapa.png"))
    plt.close(fig)

    # ---------- Figura 2: esforco e qualidade ----------
    nomes = [e["algoritmo"] for e in experimentos]
    cores = []
    for n in nomes:
        if n == "Dijkstra":
            cores.append("#7f8c8d")
        elif n == "A* (h geometrica)":
            cores.append("#1a7f37")
        elif n == "A* (h* perfeita)":
            cores.append("#16a085")
        else:
            cores.append("#2980b9")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.7), dpi=200)

    exp = [e["nos_expandidos"] for e in experimentos]
    a1.bar(range(len(nomes)), exp, color=cores)
    a1.set_yscale("log")
    a1.set_xticks(range(len(nomes)))
    a1.set_xticklabels(nomes, rotation=40, ha="right", fontsize=6.2)
    a1.set_ylabel("nós expandidos (escala log)", fontsize=7.5)
    a1.set_title("Esforço de busca", fontsize=8.6)
    a1.grid(axis="y", alpha=0.25, lw=0.5)
    for i, v in enumerate(exp):
        a1.annotate(f"{v:,}".replace(",", "."), (i, v), ha="center", va="bottom",
                    fontsize=5.8, textcoords="offset points", xytext=(0, 2))

    gaps = [e["gap_sobre_otimo_pct"] for e in experimentos]
    gaps_plot = [0.0 if g is None else g for g in gaps]
    a2.bar(range(len(nomes)), gaps_plot, color=cores)
    a2.set_xticks(range(len(nomes)))
    a2.set_xticklabels(nomes, rotation=40, ha="right", fontsize=6.2)
    a2.set_ylabel("gap sobre o custo ótimo (%)", fontsize=7.5)
    a2.set_title("Qualidade da solução", fontsize=8.6)
    a2.grid(axis="y", alpha=0.25, lw=0.5)
    for i, v in enumerate(gaps_plot):
        a2.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=5.8,
                    textcoords="offset points", xytext=(0, 2))

    fig.suptitle("Figura 2 - Comparação entre os algoritmos de busca testados",
                 fontsize=8.8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(DIR_FIGURAS, "proj1_fig2_comparacao.png"))
    plt.close(fig)


if __name__ == "__main__":
    executar()