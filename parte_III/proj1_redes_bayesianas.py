#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 1 - REDES BAYESIANAS
=============================
Diagnostico de falha de um servidor hospedeiro (host server) a partir de
metricas de monitoramento de CPU, com Rede Bayesiana construida a partir das
probabilidades publicadas em Alkasem et al. (2017), "A New Theoretical
Approach: A Model Construct for Fault Troubleshooting in Cloud Computing".

MOTIVACAO
  Uma falha de inicializacao de servidor raramente e precedida por um unico
  indicador. A combinacao de uso de CPU fisica, uso de CPU por maquinas
  virtuais e tempo em que uma VM ficou pronta porem sem ser escalonada
  (VM-CPU-Ready-Time) aumenta o risco de degradacao. Este projeto estima a
  probabilidade de falha a partir dessas evidencias usando uma Rede Bayesiana
  (uma "Naive Bayes Classifier", NBC, como no artigo de referencia).

ESTRUTURA DA REDE (DAG - grafo direcionado aciclico)
      host-CPU-usage (No/Yes) -------,
      VM-CPU-usage  (No/Yes) -------- +--> CPU-utilization (Normal/Minor/Serious)
      VM-CPU-Ready-Time (No/Yes) ----'              |
                                                    v
                                        host-server-fault (No/Yes)

  - As tres causas-raiz sao independentes a priori.
  - A categoria latente "CPU-utilization" condensa as metricas (CPT da
    Tabela 10 do artigo).
  - O sintoma "host-server-fault" depende apenas da categoria de CPU (CPT
    derivada da regra de decisao da Tabela 7 do artigo).

FONTES DOS PARAMETROS (Alkasem et al., 2017)
  - prior P(host-CPU-usage=Yes) = 0.88 .............. Tabela 9 do artigo.
  - CPT de CPU-utilization ......................... Tabela 10 do artigo.
  - priors de VM-CPU-usage / VM-CPU-Ready-Time e a CPT do no de falha sao
    HIPOTESES EXPERIMENTAIS explicitas, ajustaveis via menu/argumentos.

ALGORITMOS IMPLEMENTADOS (do zero, sem bibliotecas de rede bayesiana)
  1. Enumeracao da distribuicao conjunta  - produto de todos os fatores.
  2. Eliminacao de variavel (sum-product) - multiplica fatores e marginaliza
     as variaveis ocultas, evitando materializar a joint completa.
  As duas sao validadas uma contra a outra (devem dar o mesmo resultado).

EXECUCAO (forma principal: prompt interativo no terminal)
  python proj1_redes_bayesianas.py
  Ao iniciar, o programa roda a suite de cenarios (gerando figuras e valores
  para o relatorio) e cai num menu interativo para fixar evidencias e alterar
  parametros. Flags opcionais (ver parse_args) permitem execucao nao interativa.

DEPENDENCIAS
  numpy (arrays), matplotlib (figuras), networkx (desenho do DAG).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ===========================================================================
# 1. CONFIGURACAO DO EXPERIMENTO
# ===========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Rotulos dos estados de cada no da rede.
ESTADOS: dict[str, list[str]] = {
    "host-CPU-usage": ["No", "Yes"],
    "VM-CPU-usage": ["No", "Yes"],
    "VM-CPU-Ready-Time": ["No", "Yes"],
    "CPU-utilization": ["Normal", "Minor", "Serious"],
    "host-server-fault": ["No", "Yes"],
}

# Ordem topologica dos nos (usada para eliminar variaveis ocultas).
ORDEM_NOS = [
    "host-CPU-usage",
    "VM-CPU-usage",
    "VM-CPU-Ready-Time",
    "CPU-utilization",
    "host-server-fault",
]

# ---------------------------------------------------------------------------
# 2. CLASSE FACTOR - potencial sobre um conjunto de variaveis discretas
# ---------------------------------------------------------------------------


class Factor:
    """Distribuicao (potencial) sobre um conjunto de variaveis discretas.

    ``variaveis`` lista os nomes; ``valores`` e um array numpy cuja forma e
    (card[v] for v in variaveis), na mesma ordem da lista (a ultima variavel
    da lista e o eixo que varia "mais rapido", convencao row-major).
    """

    def __init__(self, variaveis: list[str], estados: dict[str, list[str]],
                 valores) -> None:
        self.variaveis = list(variaveis)
        self.estados = estados  # dicionario global {no: [rotulos]}
        self.card = {v: len(estados[v]) for v in self.variaveis}
        self.valores = np.asarray(valores, dtype=float)

    def _indice_estado(self, var: str, estado) -> int:
        """Converte um rotulo (ou indice) de estado para o indice numerico."""
        if isinstance(estado, (int, np.integer)):
            return int(estado)
        return self.estados[var].index(estado)

    def _alinhar(self, novas_vars: list[str]) -> np.ndarray:
        """Rearranja os eixos para a ordem de ``novas_vars`` (superset das
        proprias variaveis) inserindo eixos de tamanho 1 nas variaveis
        ausentes, permitindo multiplicacao por broadcast."""
        n = len(novas_vars)
        forma = [self.card[v] if v in self.variaveis else 1 for v in novas_vars]
        # Aumenta para ndim = n acrescentando eixos unitarios ao final.
        valores = self.valores
        if valores.ndim < n:
            valores = valores.reshape(valores.shape + (1,) * (n - valores.ndim))
        # Permutacao dos eixos: o eixo de resultado na posicao ``p`` (posicao da
        # variavel em novas_vars) recebe o eixo de origem ``j`` da propria variavel.
        axes: list[int | None] = [None] * n
        for j, v in enumerate(self.variaveis):
            axes[novas_vars.index(v)] = j
        restantes_fonte = list(range(len(self.variaveis), n))
        restantes_alvo = [p for p in range(n) if axes[p] is None]
        for p, j in zip(restantes_alvo, restantes_fonte):
            axes[p] = j
        valores = np.transpose(valores, axes)
        return valores.reshape(forma)

    def produto(self, outro: "Factor") -> "Factor":
        """Multiplica dois fatores alinhando os eixos pelo nome das variaveis."""
        novas_vars = list(dict.fromkeys(self.variaveis + outro.variaveis))
        a = self._alinhar(novas_vars)
        b = outro._alinhar(novas_vars)
        return Factor(novas_vars, self.estados, a * b)

    def marginalizar(self, var: str) -> "Factor":
        """Soma sobre todos os estados de ``var`` (elimina a variavel)."""
        idx = self.variaveis.index(var)
        novas_vars = [v for v in self.variaveis if v != var]
        return Factor(novas_vars, self.estados, np.sum(self.valores, axis=idx))

    def reduzir(self, evidencia: dict[str, object]) -> "Factor":
        """Condiciona nas variaveis de ``evidencia`` (fatiando o array)."""
        fatias: list[object] = []
        novas_vars: list[str] = []
        for v in self.variaveis:
            if v in evidencia:
                fatias.append(self._indice_estado(v, evidencia[v]))
            else:
                fatias.append(slice(None))
                novas_vars.append(v)
        return Factor(novas_vars, self.estados, self.valores[tuple(fatias)])

    def normalizar(self) -> "Factor":
        """Divide pela soma total, transformando o potencial em distribuicao."""
        total = self.valores.sum()
        if total == 0:
            raise ValueError("fator com soma zero nao pode ser normalizado")
        return Factor(self.variaveis, self.estados, self.valores / total)

    def atribuicoes(self):
        """Itera sobre (rotulos..., probabilidade) em ordem row-major."""
        for idx in np.ndindex(*self.valores.shape):
            rotulos = tuple(self.estados[v][i] for v, i in zip(self.variaveis, idx))
            yield rotulos, float(self.valores[idx])

    def prob(self, **atribuicao) -> float:
        """Devolve a probabilidade de uma atribuicao parcial dos estados."""
        indices: list[int] = []
        for v in self.variaveis:
            if v in atribuicao:
                indices.append(self._indice_estado(v, atribuicao[v]))
            else:
                indices.append(slice(None))
        return float(self.valores[tuple(indices)])


def montar_cpt(variaveis: list[str], estados: dict[str, list[str]],
               tabela: list[list[float]]) -> Factor:
    """Monta uma CPT a partir de uma tabela linha-por-estado-do-filho.

    ``variaveis`` = [pais..., filho] (o filho e o ultimo). ``tabela`` tem uma
    linha para cada estado do filho; cada linha tem prod(card dos pais)
    entradas, achatadas em row-major (ultimo pai varia mais rapido).
    """
    filho = variaveis[-1]
    shape_pais = [len(estados[v]) for v in variaveis[:-1]]
    arr = np.zeros((*shape_pais, len(estados[filho])))
    for s, linha in enumerate(tabela):
        if shape_pais:
            arr[..., s] = np.asarray(linha, dtype=float).reshape(shape_pais)
        else:
            arr[s] = float(linha[0])
    # Renormaliza cada coluna (configuracao de pais) para somar 1. Necessario
    # porque a Tabela 10 do artigo tem uma coluna que soma 1.01 (arredondamento
    # de 2 casas decimais na publicacao original).
    arr = arr / arr.sum(axis=-1, keepdims=True)
    return Factor(variaveis, estados, arr)


# ---------------------------------------------------------------------------
# 3. CLASSE REDE BAYESIANA
# ---------------------------------------------------------------------------


class RedeBayesiana:
    """Grafo direcionado aciclico com CPTs associadas a cada no."""

    def __init__(self, estados: dict[str, list[str]]) -> None:
        self.estados = estados
        self.arcos: list[tuple[str, str]] = []
        self.cpts: dict[str, Factor] = {}

    def adicionar_arco(self, pai: str, filho: str) -> None:
        if (filho, pai) in self.arcos:
            raise ValueError(f"arco {pai}->{filho} criaria um ciclo")
        if (pai, filho) not in self.arcos:
            self.arcos.append((pai, filho))

    def pais(self, no: str) -> list[str]:
        return [p for p, f in self.arcos if f == no]

    def validar(self) -> bool:
        """Confere se e um DAG e se cada CPT soma 1 para cada configuracao de pais."""
        # Verifica aciclicidade por ordenacao topologica (Kahn).
        grau = {v: 0 for v in self.estados}
        adj = {v: [] for v in self.estados}
        for p, f in self.arcos:
            adj[p].append(f)
            grau[f] += 1
        ordem = []
        fila = [v for v in grau if grau[v] == 0]
        while fila:
            v = fila.pop()
            ordem.append(v)
            for f in adj[v]:
                grau[f] -= 1
                if grau[f] == 0:
                    fila.append(f)
        if len(ordem) != len(self.estados):
            raise ValueError("o grafo possui um ciclo (nao e um DAG)")

        for no, fator in self.cpts.items():
            # Soma sobre o proprio no; cada configuracao de pais deve dar 1.
            marg = fator.marginalizar(no)
            if not np.allclose(marg.valores, 1.0, atol=1e-6):
                raise ValueError(f"CPT de '{no}' nao soma 1 em alguma configuracao")
        return True

    def fatores(self) -> list[Factor]:
        return list(self.cpts.values())

    # --- Inferencia exata 1: enumeracao da distribuicao conjunta ---

    def distribuicao_conjunta(self) -> Factor:
        """Produto de todos os fatores -> P(todas as variaveis)."""
        fatores = self.fatores()
        joint = fatores[0]
        for f in fatores[1:]:
            joint = joint.produto(f)
        return joint

    def enumerar(self, consulta: list[str],
                 evidencia: dict[str, object]) -> Factor:
        """P(consulta | evidencia) por forca bruta sobre a joint."""
        joint = self.distribuicao_conjunta()
        if evidencia:
            joint = joint.reduzir(evidencia)
        for v in list(joint.variaveis):
            if v not in consulta:
                joint = joint.marginalizar(v)
        return joint.normalizar()

    # --- Inferencia exata 2: eliminacao de variavel (sum-product) ---

    def eliminacao_variavel(self, consulta: list[str],
                            evidencia: dict[str, object]) -> Factor:
        """P(consulta | evidencia) somando as variaveis ocultas uma a uma."""
        fatores = [f.reduzir(evidencia) if evidencia else f for f in self.fatores()]
        ocultas = [v for v in ORDEM_NOS if v not in consulta and v not in evidencia]
        for var in ocultas:
            relevantes = [f for f in fatores if var in f.variaveis]
            if not relevantes:
                continue
            fatores = [f for f in fatores if var not in f.variaveis]
            prod = relevantes[0]
            for f in relevantes[1:]:
                prod = prod.produto(f)
            fatores.append(prod.marginalizar(var))
        resultado = fatores[0]
        for f in fatores[1:]:
            resultado = resultado.produto(f)
        return resultado.normalizar()


# ---------------------------------------------------------------------------
# 4. CONSTRUCAO DA REDE (parametros do artigo + hipoteses ajustaveis)
# ---------------------------------------------------------------------------


def construir_rede(prior_host_cpu_yes: float = 0.88,
                   prior_vm_cpu_yes: float = 0.30,
                   prior_vm_ready_yes: float = 0.35,
                   p_falha_dado_normal: float = 0.05,
                   p_falha_dado_minor: float = 0.20,
                   p_falha_dado_serious: float = 0.90) -> RedeBayesiana:
    """Constroi a rede do submodelo de CPU com as CPTs do artigo.

    Todos os valores de probabilidade sao parametros, o que permite o estudo
    de sensibilidade (ex.: servidor "confiavel" com prior menor de falha).
    """
    rede = RedeBayesiana(ESTADOS)

    rede.adicionar_arco("host-CPU-usage", "CPU-utilization")
    rede.adicionar_arco("VM-CPU-usage", "CPU-utilization")
    rede.adicionar_arco("VM-CPU-Ready-Time", "CPU-utilization")
    rede.adicionar_arco("CPU-utilization", "host-server-fault")

    # --- Priors das causas-raiz (hipoteses explicitas) ---
    rede.cpts["host-CPU-usage"] = montar_cpt(
        ["host-CPU-usage"], ESTADOS,
        [[1 - prior_host_cpu_yes], [prior_host_cpu_yes]])
    rede.cpts["VM-CPU-usage"] = montar_cpt(
        ["VM-CPU-usage"], ESTADOS,
        [[1 - prior_vm_cpu_yes], [prior_vm_cpu_yes]])
    rede.cpts["VM-CPU-Ready-Time"] = montar_cpt(
        ["VM-CPU-Ready-Time"], ESTADOS,
        [[1 - prior_vm_ready_yes], [prior_vm_ready_yes]])

    # --- CPT de CPU-utilization (Tabela 10 do artigo) ---
    # Ordem dos pais: [VM-CPU-Ready-Time, host-CPU-usage, VM-CPU-usage].
    # Cada linha e um estado do filho (Normal, Minor, Serious); as colunas
    # seguem a ordem da Tabela 10 (VM-CPU-usage varia mais rapido).
    tabela_cpu = [
        [0.17, 0.23, 0.11, 0.12, 0.19, 0.13, 0.21, 0.11],  # Normal
        [0.52, 0.13, 0.32, 0.25, 0.14, 0.15, 0.14, 0.31],  # Minor
        [0.31, 0.64, 0.58, 0.63, 0.67, 0.72, 0.65, 0.58],  # Serious
    ]
    rede.cpts["CPU-utilization"] = montar_cpt(
        ["VM-CPU-Ready-Time", "host-CPU-usage", "VM-CPU-usage", "CPU-utilization"],
        ESTADOS, tabela_cpu)

    # --- CPT de host-server-fault (derivada da Tabela 7, suavizada) ---
    # Regra do artigo: Normal/Minor -> "no", Serious -> "yes"; aqui a regra e
    # probabilistica para nao ser deterministica.
    rede.cpts["host-server-fault"] = montar_cpt(
        ["CPU-utilization", "host-server-fault"], ESTADOS,
        [
            [1 - p_falha_dado_normal, 1 - p_falha_dado_minor, 1 - p_falha_dado_serious],
            [p_falha_dado_normal, p_falha_dado_minor, p_falha_dado_serious],
        ])

    return rede


# ---------------------------------------------------------------------------
# 5. CONSULTA E COMPARACAO ENTRE OS DOIS METODOS
# ---------------------------------------------------------------------------


def consultar(rede: RedeBayesiana, alvo: str, evidencia: dict[str, object]):
    """Calcula P(alvo | evidencia) pelos dois metodos e confere a igualdade."""
    por_enum = rede.enumerar([alvo], evidencia)
    por_elim = rede.eliminacao_variavel([alvo], evidencia)
    if not np.allclose(por_enum.valores, por_elim.valores, atol=1e-9):
        raise RuntimeError("divergencia entre enumeracao e eliminacao de variavel")
    return por_elim


def imprime_distribuicao(fator: Factor, titulo: str = "") -> None:
    """Imprime um fator univariado como tabela de probabilidades."""
    if titulo:
        print(f"\n{titulo}")
    for rotulos, p in fator.atribuicoes():
        print(f"    {rotulos[0]:<8} = {p:.4f}")


# ---------------------------------------------------------------------------
# 6. CENARIOS (suite pre-definida) E SENSIBILIDADE
# ---------------------------------------------------------------------------


def definir_cenarios() -> list[tuple[str, dict[str, str]]]:
    """Devolve os cenarios de evidencia usados no relatorio."""
    return [
        ("1. Operacao normal (sem evidencia)", {}),
        ("2. CPU fisica saturada", {"host-CPU-usage": "Yes"}),
        ("3. VM consumindo CPU", {"VM-CPU-usage": "Yes"}),
        ("4. VM aguardando escalonar", {"VM-CPU-Ready-Time": "Yes"}),
        ("5. Saturação critica (todas)", {
            "host-CPU-usage": "Yes",
            "VM-CPU-usage": "Yes",
            "VM-CPU-Ready-Time": "Yes"}),
        ("6. Evidencia negativa", {"host-CPU-usage": "No"}),
        ("7. CPU classificada como Serious", {"CPU-utilization": "Serious"}),
        ("8. CPU classificada como Normal", {"CPU-utilization": "Normal"}),
    ]


def rodar_cenarios(rede: RedeBayesiana) -> list[dict]:
    """Roda todos os cenarios e devolve os resultados estruturados."""
    resultados = []
    prior = consultar(rede, "host-server-fault", {})
    p_falha_prior = prior.prob(**{"host-server-fault": "Yes"})
    print("=" * 70)
    print("RESULTADOS DA SUITE DE CENARIOS")
    print(f"P(host-server-fault=Yes) a priori (sem evidencia) = {p_falha_prior:.4f}")
    print("=" * 70)

    for nome, evidencia in definir_cenarios():
        p_falha = consultar(rede, "host-server-fault", evidencia)
        p_falha_yes = p_falha.prob(**{"host-server-fault": "Yes"})
        # Quando a propria categoria de CPU e a evidencia, a posterior de
        # CPU-utilization e degenerada; monta-se diretamente do rotulo.
        if "CPU-utilization" in evidencia:
            rotulo = evidencia["CPU-utilization"]
            p_cpu = {"Normal": 1.0 if rotulo == "Normal" else 0.0,
                     "Minor": 1.0 if rotulo == "Minor" else 0.0,
                     "Serious": 1.0 if rotulo == "Serious" else 0.0}
        else:
            fator_cpu = consultar(rede, "CPU-utilization", evidencia)
            p_cpu = {rotulos[0]: p for rotulos, p in fator_cpu.atribuicoes()}
        resultados.append({
            "cenario": nome,
            "evidencia": evidencia or {},
            "p_falha_yes": p_falha_yes,
            "p_falha_prior": p_falha_prior,
            "p_cpu_normal": p_cpu["Normal"],
            "p_cpu_minor": p_cpu["Minor"],
            "p_cpu_serious": p_cpu["Serious"],
        })
        ev_str = ", ".join(f"{k}={v}" for k, v in evidencia.items()) or "-"
        print(f"\n{nome}")
        print(f"    evidencias: {ev_str}")
        print(f"    P(host-server-fault=Yes | ev) = {p_falha_yes:.4f}  (prior = {p_falha_prior:.4f})")
        print("    P(CPU-utilization | ev): "
              + "  ".join(f"{k}={v:.4f}" for k, v in p_cpu.items()))
    return resultados


def sensibilidade_confiabilidade(rede_base_params: dict,
                                 de: float = 0.30, ate: float = 1.00,
                                 n: int = 11) -> dict:
    """Varre a CPT de falha (p_falha_dado_serious) representando servidores
    com diferentes niveis de confiabilidade e registra P(falha) sem evidencia.

    Um servidor "robusto" converte CPU critica em falha com baixa probabilidade
    (p_falha_dado_serious pequeno); um servidor "fragil" converte com alta
    probabilidade. Varre-se esse parametro mantendo os demais fixos.
    """
    xs = np.linspace(de, ate, n)
    sem_ev = []
    com_ev = []
    for p in xs:
        params = dict(rede_base_params)
        params["p_falha_dado_serious"] = float(p)
        rede = construir_rede(**params)
        sem_ev.append(consultar(rede, "host-server-fault", {})
                      .prob(**{"host-server-fault": "Yes"}))
        com_ev.append(consultar(rede, "host-server-fault",
                                {"CPU-utilization": "Serious"})
                      .prob(**{"host-server-fault": "Yes"}))
    return {"p_falha_dado_serious": xs.tolist(),
            "sem_evidencia": sem_ev, "com_evidencia": com_ev}


def diagnostico_falha(rede: RedeBayesiana) -> dict:
    """Reproduz a propagacao diagnostica do artigo (Secao 5, Figura 7).

    Dado que o servidor apresentou falha (host-server-fault = Yes), calcula a
    probabilidade de cada causa-raiz e da categoria de CPU, identificando a
    causa mais provavel (no artigo: host-CPU-usage ~ 88%).
    """
    alvos = ["host-CPU-usage", "VM-CPU-usage", "VM-CPU-Ready-Time", "CPU-utilization"]
    resultado: dict = {"evidencia": {"host-server-fault": "Yes"}, "causas": {}}
    print("\n" + "=" * 70)
    print("DIAGNOSTICO — P(causa | host-server-fault = Yes)")
    print("=" * 70)
    for alvo in alvos:
        fator = consultar(rede, alvo, {"host-server-fault": "Yes"})
        dist = {rotulos[0]: round(p, 4) for rotulos, p in fator.atribuicoes()}
        resultado["causas"][alvo] = dist
        linha = "  " + "  ".join(f"{k}={v:.4f}" for k, v in dist.items())
        print(f"  P({alvo} | falha) -> {linha}")
    return resultado


# ---------------------------------------------------------------------------
# 7. VISUALIZACAO
# ---------------------------------------------------------------------------


def desenhar_dag(rede: RedeBayesiana, caminho: str) -> None:
    """Desenha a estrutura do DAG em camadas."""
    G = nx.DiGraph()
    G.add_nodes_from(rede.estados.keys())
    G.add_edges_from(rede.arcos)

    camadas = {
        "host-CPU-usage": 0, "VM-CPU-usage": 0, "VM-CPU-Ready-Time": 0,
        "CPU-utilization": 1, "host-server-fault": 2,
    }
    pos = {}
    for no, camada in camadas.items():
        idx = {"host-CPU-usage": 0, "VM-CPU-usage": 1, "VM-CPU-Ready-Time": 2}.get(no)
        if camada == 0:
            pos[no] = (0, -idx)
        elif camada == 1:
            pos[no] = (1, -1)
        else:
            pos[no] = (2, -1)

    cores = ["#c7e9b4", "#7fcdbb", "#2c7fb8"]
    node_color = [cores[camadas[n]] for n in G.nodes()]

    plt.figure(figsize=(7, 4))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>",
                           arrowsize=18, node_size=0, width=1.5, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=2600)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("Estrutura da Rede Bayesiana (DAG) — submodelo de CPU")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


def plot_priori_posteriori(resultados: list[dict], caminho: str) -> None:
    """Grafico de barras: P(falha) por cenario + distribuicao de CPU."""
    nomes = [r["cenario"].split(". ", 1)[1] for r in resultados]
    p_falha = [r["p_falha_yes"] for r in resultados]
    prior = resultados[0]["p_falha_prior"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    x = np.arange(len(nomes))
    ax1.bar(x, p_falha, color="#2c7fb8", alpha=0.85)
    ax1.axhline(prior, color="#d7191c", linestyle="--", linewidth=1.5,
                label=f"prior = {prior:.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("P(host-server-fault = Yes)")
    ax1.set_title("Probabilidade de falha por cenario (a posteriori)")
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    for i, v in enumerate(p_falha):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    p_normal = [r["p_cpu_normal"] for r in resultados]
    p_minor = [r["p_cpu_minor"] for r in resultados]
    p_serious = [r["p_cpu_serious"] for r in resultados]
    ax2.bar(x, p_normal, label="Normal", color="#c7e9b4")
    ax2.bar(x, p_minor, bottom=p_normal, label="Minor", color="#7fcdbb")
    ax2.bar(x, p_serious, bottom=np.array(p_normal) + np.array(p_minor),
            label="Serious", color="#2c7fb8")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nomes, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("P(CPU-utilization)")
    ax2.set_title("Distribuicao da categoria de CPU por cenario")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sensibilidade(sens: dict, caminho: str) -> None:
    """Curva de P(falha) em funcao da confiabilidade (p_falha_dado_serious)."""
    plt.figure(figsize=(7, 4))
    xs = sens["p_falha_dado_serious"]
    plt.plot(xs, sens["sem_evidencia"], marker="o", label="sem evidencia (risco basal)")
    plt.plot(xs, sens["com_evidencia"], marker="s",
             label="com evidencia CPU-utilization=Serious")
    plt.xlabel("confiabilidade: P(host-server-fault=Yes | CPU=Serious)")
    plt.ylabel("P(host-server-fault = Yes)")
    plt.title("Sensibilidade da confiabilidade do servidor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. MENU INTERATIVO (forma principal de execucao)
# ---------------------------------------------------------------------------


def menu_interativo(rede: RedeBayesiana, params: dict) -> None:
    """Loop interativo: o usuario fixa evidencias e altera parametros."""
    evidencia: dict[str, str] = {}
    print("\n" + "=" * 70)
    print("MENU INTERATIVO — diagnostico de falha do servidor")
    print("Digite uma opcao e ENTER (ou 'q' para sair, '?' para ajuda).")
    print("=" * 70)

    def mostrar():
        p_falha = consultar(rede, "host-server-fault", evidencia)
        p_cpu = consultar(rede, "CPU-utilization", evidencia)
        ev_str = ", ".join(f"{k}={v}" for k, v in evidencia.items()) or "(nenhuma)"
        print(f"\nEvidencias atuais: {ev_str}")
        print(f"  P(host-server-fault=Yes | ev) = "
              f"{p_falha.prob(**{'host-server-fault': 'Yes'}):.4f}")
        print(f"  P(CPU-utilization=Serious | ev) = "
              f"{p_cpu.prob(**{'CPU-utilization': 'Serious'}):.4f}")

    def ajuda():
        print("\nComandos:")
        print("  h -> alternar host-CPU-usage   (No/Yes/nao observar)")
        print("  c -> alternar VM-CPU-usage     (No/Yes/nao observar)")
        print("  r -> alternar VM-CPU-Ready-Time(No/Yes/nao observar)")
        print("  l -> limpar todas as evidencias")
        print("  p -> alterar prior de host-CPU-usage (ex.: 0.50)")
        print("  ? -> ajuda")
        print("  q -> sair")

    def alternar(var: str):
        if var not in evidencia:
            evidencia[var] = "Yes"
        elif evidencia[var] == "Yes":
            evidencia[var] = "No"
        else:
            del evidencia[var]

    mostrar()
    while True:
        try:
            op = input("\n>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if op == "q":
            break
        if op == "?":
            ajuda()
            continue
        if op == "l":
            evidencia.clear()
        elif op in ("h", "c", "r"):
            mapa = {"h": "host-CPU-usage", "c": "VM-CPU-usage",
                    "r": "VM-CPU-Ready-Time"}
            alternar(mapa[op])
        elif op == "p":
            try:
                novo = float(input("  prior P(host-CPU-usage=Yes) [0.0-1.0]: "))
            except ValueError:
                print("  valor invalido")
                continue
            if not (0.0 <= novo <= 1.0):
                print("  valor deve estar em [0,1]")
                continue
            params["prior_host_cpu_yes"] = novo
            rede = construir_rede(**params)
            print(f"  prior atualizado para {novo:.2f}")
        else:
            print("  comando desconhecido (digite 'h' para ajuda)")
            continue
        mostrar()


# ---------------------------------------------------------------------------
# 9. PERSISTENCIA DOS RESULTADOS
# ---------------------------------------------------------------------------


def salvar_resultados(resultados: list[dict], sens: dict, diagnostico: dict) -> None:
    """Grava JSON e CSV com os valores numericos dos experimentos."""
    caminho_json = os.path.join(DIR_RESULTADOS, "proj1_resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump({"cenarios": resultados, "sensibilidade": sens,
                   "diagnostico": diagnostico}, fh, ensure_ascii=False, indent=2)

    caminho_csv = os.path.join(DIR_RESULTADOS, "proj1_tabela.csv")
    cabecalho = ["cenario", "p_falha_yes", "p_falha_prior",
                 "p_cpu_normal", "p_cpu_minor", "p_cpu_serious"]
    with open(caminho_csv, "w", encoding="utf-8") as fh:
        fh.write(";".join(cabecalho) + "\n")
        for r in resultados:
            linha = [r["cenario"], f"{r['p_falha_yes']:.4f}",
                     f"{r['p_falha_prior']:.4f}", f"{r['p_cpu_normal']:.4f}",
                     f"{r['p_cpu_minor']:.4f}", f"{r['p_cpu_serious']:.4f}"]
            fh.write(";".join(linha) + "\n")


# ---------------------------------------------------------------------------
# 10. ARGUMENTOS DE LINHA DE COMANDO (opcionais)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rede Bayesiana para diagnostico de falha de servidor")
    parser.add_argument("--evidence", type=str, default=None,
                        help="evidencias no formato 'host-CPU-usage=Yes,VM-CPU-usage=Yes'")
    parser.add_argument("--prior-host-cpu", type=float, default=None,
                        help="prior P(host-CPU-usage=Yes) (default 0.88)")
    parser.add_argument("--nao-interativo", action="store_true",
                        help="nao abre o menu interativo (roda apenas cenarios)")
    return parser.parse_args()


def interpretar_evidencia(texto: str) -> dict[str, str]:
    evidencia: dict[str, str] = {}
    for parte in texto.split(","):
        if "=" not in parte:
            continue
        k, v = parte.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in ESTADOS:
            raise ValueError(f"no desconhecido: {k}")
        if v not in ESTADOS[k]:
            raise ValueError(f"estado invalido para {k}: {v} (valido: {ESTADOS[k]})")
        evidencia[k] = v
    return evidencia


# ---------------------------------------------------------------------------
# 11. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    params = {
        "prior_host_cpu_yes": 0.88,
        "prior_vm_cpu_yes": 0.30,
        "prior_vm_ready_yes": 0.35,
        "p_falha_dado_normal": 0.05,
        "p_falha_dado_minor": 0.20,
        "p_falha_dado_serious": 0.90,
    }
    if args.prior_host_cpu is not None:
        params["prior_host_cpu_yes"] = args.prior_host_cpu

    rede = construir_rede(**params)
    rede.validar()
    print("Rede Bayesiana construida e validada (DAG + CPTs).")

    # Suite de cenarios + diagnostico + sensibilidade + persistencia.
    resultados = rodar_cenarios(rede)
    diagnostico = diagnostico_falha(rede)
    sens = sensibilidade_confiabilidade(params)
    salvar_resultados(resultados, sens, diagnostico)
    desenhar_dag(rede, os.path.join(DIR_FIGURAS, "proj1_fig1_dag.png"))
    plot_priori_posteriori(
        resultados, os.path.join(DIR_FIGURAS, "proj1_fig2_priori_posteriori.png"))
    plot_sensibilidade(
        sens, os.path.join(DIR_FIGURAS, "proj1_fig3_sensibilidade.png"))
    print("\nFiguras salvas em:", DIR_FIGURAS)
    print("Resultados salvos em:", DIR_RESULTADOS)

    # Execucao nao interativa via flag.
    if args.evidence:
        evidencia = interpretar_evidencia(args.evidence)
        p_falha = consultar(rede, "host-server-fault", evidencia)
        p_cpu = consultar(rede, "CPU-utilization", evidencia)
        print(f"\nConsulta com evidencia {evidencia}:")
        print(f"  P(host-server-fault=Yes) = {p_falha.prob(**{'host-server-fault': 'Yes'}):.4f}")
        imprime_distribuicao(p_cpu, "  P(CPU-utilization):")
        return

    if args.nao_interativo:
        return

    # Forma principal: menu interativo no terminal.
    menu_interativo(rede, params)


if __name__ == "__main__":
    main()
