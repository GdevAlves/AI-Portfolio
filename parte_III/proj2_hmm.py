#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 2 - MODELOS MARKOVIANOS OCULTOS (HMM)
==============================================
Monitoramento da condicao operacional de uma maquina industrial a partir de
sinais de sensores (temperatura, vibracao e ruido), usando um HMM com
probabilidades definidas manualmente (sem treinamento via Baum-Welch).

MOTIVACAO
  O estado real da maquina (condicao de saude) nao e diretamente observavel:
  os sensores fornecem apenas sinais que SUGEREM determinadas condicoes. Uma
  temperatura elevada pode indicar superaquecimento, mas tambem ocorre em uma
  maquina apenas desgastada. Um HMM estima a sequencia mais provavel de estados
  ocultos ao longo do tempo, combinando as observacoes e as probabilidades de
  transicao entre estados.

ESTADOS OCULTOS (condicao real da maquina)
  0 = Normal            - funcionamento adequado
  1 = Desgaste          - deterioracao
  2 = Superaquecimento  - temperatura critica
  3 = Falha             - falha operacional

OBSERVACOES (sinais dos sensores)
  0 = Temperatura normal
  1 = Temperatura elevada
  2 = Vibracao elevada
  3 = Ruido anormal

COMPONENTES DO HMM
  pi  = probabilidades iniciais (P(q_0))
  A   = matriz de transicao (P(q_t | q_{t-1}))  - linhas somam 1
  B   = matriz de emissao  (P(o_t | q_t))       - linhas somam 1

ALGORITMOS IMPLEMENTADOS (do zero, com NumPy)
  1. Forward (com escalonamento) -> alfa_t(i) = P(o_1..o_t, q_t=i), que
     normalizado da a posteriori de FILTRAGEM P(q_t | o_1..o_t), alem da
     log-verossimilhanca da sequencia (soma dos logs dos fatores de escala).
  2. Viterbi (em log-probabilidades) -> caminho mais provavel de estados e a
     probabilidade desse caminho.

EXECUCAO (forma principal: prompt interativo no terminal)
  python proj2_hmm.py
  Ao iniciar, roda a suite de experimentos (gerando figuras e valores para o
  relatorio) e cai num menu interativo para digitar sequencias de observacoes
  (codigos: n/t/v/r). Flags opcionais: --sequencia, --nao-interativo.

DEPENDENCIAS
  numpy (matrizes), matplotlib (figuras).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# ===========================================================================
# 1. CONFIGURACAO
# ===========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)

ESTADOS = ["Normal", "Desgaste", "Superaquecimento", "Falha"]
OBSERVACOES = ["Temperatura normal", "Temperatura elevada",
               "Vibracao elevada", "Ruido anormal"]

# Codigo digitado no terminal -> indice da observacao.
CODIGOS = {"n": 0, "t": 1, "v": 2, "r": 3}
CODIGO_INVERSO = {0: "n", 1: "t", 2: "v", 3: "r"}

# Cores dos estados (usadas nas figuras).
CORES_ESTADO = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]

# ---------------------------------------------------------------------------
# 2. MODELO: probabilidades iniciais, de transicao e de emissao
# ---------------------------------------------------------------------------

# Probabilidades iniciais: a maquina normalmente inicia saudavel.
PI = np.array([0.80, 0.15, 0.04, 0.01])

# Matriz de transicao padrao (linhas = estado atual, colunas = proximo).
A_PADRAO = np.array([
    [0.85, 0.10, 0.04, 0.01],   # Normal
    [0.20, 0.70, 0.08, 0.02],   # Desgaste
    [0.05, 0.10, 0.75, 0.10],   # Superaquecimento
    [0.30, 0.20, 0.10, 0.40],   # Falha
])

# Variantes de A para o experimento 4 (alteracao das transicoes).
A_RIGIDA = np.array([           # alta permanencia (degradacao lenta)
    [0.95, 0.04, 0.01, 0.00],
    [0.10, 0.85, 0.04, 0.01],
    [0.02, 0.05, 0.90, 0.03],
    [0.05, 0.10, 0.05, 0.80],
])

A_PROGRESSIVA = np.array([      # degradacao rapida
    [0.60, 0.25, 0.10, 0.05],
    [0.10, 0.55, 0.25, 0.10],
    [0.02, 0.08, 0.60, 0.30],
    [0.30, 0.20, 0.10, 0.40],
])

# Matriz de emissao (linhas = estado, colunas = observacao). Cada estado tem
# uma "assinatura": Normal -> temperatura normal; Desgaste -> temperatura
# elevada; Superaquecimento -> temperatura elevada (mais forte); Falha ->
# vibracao/ruido.
B = np.array([
    [0.80, 0.10, 0.07, 0.03],   # Normal
    [0.30, 0.40, 0.25, 0.05],   # Desgaste
    [0.10, 0.70, 0.15, 0.05],   # Superaquecimento
    [0.05, 0.20, 0.40, 0.35],   # Falha
])


def construir_hmm(matriz_transicao: np.ndarray | None = None):
    """Devolve (pi, A, B). Permite trocar A no experimento de transicoes."""
    A = A_PADRAO if matriz_transicao is None else matriz_transicao
    return PI.copy(), A, B


# ---------------------------------------------------------------------------
# 3. ALGORITMO FORWARD (com escalonamento) e VITERBI
# ---------------------------------------------------------------------------

def forward(pi: np.ndarray, A: np.ndarray, B: np.ndarray,
            obs: list[int]) -> tuple[np.ndarray, float]:
    """Calcula a posteriori de filtragem e a log-verossimilhanca da sequencia.

    alfa_t(i) = P(o_1..o_t, q_t=i). Para evitar underflow, cada coluna t e
    normalizada por um fator de escala c_t; a log-verossimilhanca da sequencia
    e soma de log(c_t). O alfa normalizado e exatamente a posteriori de
    filtragem P(q_t | o_1..o_t).
    """
    T = len(obs)
    N = len(pi)
    alfa = np.zeros((T, N))
    escalas = np.zeros(T)

    alfa[0] = pi * B[:, obs[0]]
    escalas[0] = alfa[0].sum()
    alfa[0] /= escalas[0]

    for t in range(1, T):
        alfa[t] = (alfa[t - 1] @ A) * B[:, obs[t]]
        escalas[t] = alfa[t].sum()
        alfa[t] /= escalas[t]

    log_verossimilhanca = float(np.log(escalas).sum())
    return alfa, log_verossimilhanca


def viterbi(pi: np.ndarray, A: np.ndarray, B: np.ndarray,
            obs: list[int]) -> tuple[np.ndarray, float]:
    """Devolve o caminho mais provavel de estados e sua log-probabilidade.

    Trabalha em log para estabilidade numerica. delta_t(j) guarda a maior
    log-probabilidade de chegar ao estado j no instante t; psi_t(j) guarda o
    estado anterior dessa melhor rota. Ao final, refaz-se o caminho de tras
    para frente.
    """
    T = len(obs)
    N = len(pi)
    log_pi = np.log(pi + 1e-12)
    log_A = np.log(A + 1e-12)
    log_B = np.log(B + 1e-12)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            pontuacoes = delta[t - 1] + log_A[:, j]
            psi[t, j] = int(np.argmax(pontuacoes))
            delta[t, j] = pontuacoes[psi[t, j]] + log_B[j, obs[t]]

    caminho = np.zeros(T, dtype=int)
    caminho[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        caminho[t] = psi[t + 1, caminho[t + 1]]

    return caminho, float(delta[-1].max())


# ---------------------------------------------------------------------------
# 4. DECODIFICACAO DA ENTRADA DO USUARIO
# ---------------------------------------------------------------------------

def decodificar_sequencia(texto: str) -> list[int] | None:
    """Converte uma string de codigos (n/t/v/r) em indices de observacao."""
    texto = texto.strip().lower()
    if not texto:
        return None
    tokens = texto.split() if " " in texto else list(texto)
    seq = []
    for tk in tokens:
        if tk not in CODIGOS:
            raise ValueError(f"codigo invalido: {tk!r} (use n/t/v/r)")
        seq.append(CODIGOS[tk])
    return seq


# ---------------------------------------------------------------------------
# 5. EXPERIMENTOS
# ---------------------------------------------------------------------------

EXPERIMENTOS = [
    {"nome": "1. Funcionamento normal", "obs": "n" * 12},
    {"nome": "2. Degradacao gradual", "obs": "n" * 4 + "t" * 4 + "v" * 4},
    {"nome": "3. Falha", "obs": "rvt rvrvt rvr v".replace(" ", "")},
    {"nome": "4. Transicoes (comparacao)", "obs": "n" * 4 + "t" * 4 + "v" * 4,
     "comparar_transicoes": True},
    {"nome": "5. Observacoes ruidosas", "obs": "ntvr nvtr ntv r".replace(" ", "")},
]


def rodar_experimentos() -> list[dict]:
    """Executa todos os experimentos e devolve os resultados estruturados."""
    resultados = []
    print("=" * 70)
    print("RESULTADOS DOS EXPERIMENTOS")
    print("=" * 70)

    for exp in EXPERIMENTOS:
        obs = decodificar_sequencia(exp["obs"])

        if exp.get("comparar_transicoes"):
            print(f"\n{exp['nome']}")
            print(f"    sequencia: {' '.join(CODIGO_INVERSO[o] for o in obs)}")
            linha = {"cenario": exp["nome"],
                     "sequencia": exp["obs"], "variantes": {}}
            for rotulo, A_variante in (("padrao", A_PADRAO),
                                       ("rigida", A_RIGIDA),
                                       ("progressiva", A_PROGRESSIVA)):
                pi, A, Bm = construir_hmm(A_variante)
                caminho, logp = viterbi(pi, A, Bm, obs)
                _, loglik = forward(pi, A, Bm, obs)
                estados = " -> ".join(ESTADOS[c] for c in caminho)
                print(f"    [{rotulo:>11}] log-veross.={loglik:8.3f}  "
                      f"caminho: {estados}")
                linha["variantes"][rotulo] = {
                    "log_verossimilhanca": loglik,
                    "log_prob_caminho": logp,
                    "caminho": [ESTADOS[c] for c in caminho],
                }
            resultados.append(linha)
            continue

        pi, A, Bm = construir_hmm()
        alfa, loglik = forward(pi, A, Bm, obs)
        caminho, logp = viterbi(pi, A, Bm, obs)
        resultado = {
            "cenario": exp["nome"],
            "sequencia": exp["obs"],
            "log_verossimilhanca": loglik,
            "log_prob_caminho": logp,
            "caminho": [ESTADOS[c] for c in caminho],
            "posterior": alfa.tolist(),
        }
        resultados.append(resultado)

        print(f"\n{exp['nome']}")
        print(f"    observacoes : {' '.join(CODIGO_INVERSO[o] for o in obs)}")
        print(f"    log-verossimilhanca : {loglik:.3f}")
        print("    caminho Viterbi : " + " -> ".join(ESTADOS[c] for c in caminho))

    return resultados


# ---------------------------------------------------------------------------
# 6. VISUALIZACAO
# ---------------------------------------------------------------------------

def desenhar_modelo(caminho: str) -> None:
    """Diagrama esquematico do HMM: estados (esq) e observacoes (dir)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.7, 3.7)
    ax.axis("off")

    pos_estado = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    pos_obs = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}

    # Nos de estado (círculos).
    for i, nome in enumerate(ESTADOS):
        ax.add_patch(Circle((0.0, pos_estado[i]), 0.17,
                                color=CORES_ESTADO[i], ec="black", lw=1.5))
        ax.text(0.0, pos_estado[i] + 0.32, nome, ha="center", fontsize=11)

    # Nos de observacao (retangulos).
    for i, nome in enumerate(OBSERVACOES):
        ax.add_patch(FancyBboxPatch((0.95, pos_obs[i] - 0.13), 0.55, 0.26,
                                        boxstyle="round,pad=0.02",
                                        facecolor="#eeeeee", ec="black", lw=1.2))
        ax.text(1.225, pos_obs[i], nome, ha="center", va="center", fontsize=9)

    # Transicoes entre estados (setas estruturais).
    transicoes = [(0, 1, "0.10"), (1, 2, "0.08"), (2, 3, "0.10"), (3, 0, "0.30")]
    for origem, destino, prob in transicoes:
        y0, y1 = pos_estado[origem], pos_estado[destino]
        dx = 0.17 if destino >= origem else -0.17
        ax.annotate("", xy=(0.0, y1), xytext=(dx, y0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
        ax.text(0.42, (y0 + y1) / 2, prob, fontsize=9, color="dimgray")

    # Auto-transicoes (permanencia).
    for i, (nome, prob) in enumerate(zip(ESTADOS, [0.85, 0.70, 0.75, 0.40])):
        ax.annotate("", xy=(0.13, pos_estado[i] + 0.13),
                    xytext=(-0.13, pos_estado[i] + 0.13),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.5",
                                    color="black", lw=1.2))
        ax.text(-0.30, pos_estado[i] + 0.28, str(prob), fontsize=9, color="dimgray")

    # Emissoes (seta de cada estado para sua observacao dominante).
    emissoes = [(0, 0, "0.80"), (1, 1, "0.40"), (2, 1, "0.70"), (3, 2, "0.40")]
    for estado, obs, prob in emissoes:
        ax.annotate("", xy=(0.95, pos_obs[obs]), xytext=(0.17, pos_estado[estado]),
                    arrowprops=dict(arrowstyle="->", color=CORES_ESTADO[estado],
                                    lw=1.4, alpha=0.8))
        ax.text(0.55, (pos_estado[estado] + pos_obs[obs]) / 2, prob,
                fontsize=8, color=CORES_ESTADO[estado])

    ax.text(0.0, -0.6, "Estados ocultos (q)", ha="center", fontsize=10)
    ax.text(1.225, -0.6, "Observacoes (o)", ha="center", fontsize=10)
    fig.suptitle("Modelo HMM da maquina industrial", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


def plot_experimento_detalhado(resultado: dict, caminho: str) -> None:
    """Posterior de filtragem + observacoes ao longo do tempo (1 experimento)."""
    obs = decodificar_sequencia(resultado["sequencia"])
    alfa = np.array(resultado["posterior"])
    t = np.arange(len(obs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.step(t, obs, where="mid", color="black", lw=1.5)
    ax1.set_yticks(range(len(OBSERVACOES)))
    ax1.set_yticklabels([f"{i}={c}" for i, c in
                         [(0, 'n'), (1, 't'), (2, 'v'), (3, 'r')]], fontsize=10)
    ax1.set_ylabel("Observacao", fontsize=10)
    ax1.set_title("Observacoes dos sensores", fontsize=11)

    for i, nome in enumerate(ESTADOS):
        ax2.plot(t, alfa[:, i], marker="o", label=nome, color=CORES_ESTADO[i], lw=1.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("P(estado | observacoes)", fontsize=10)
    ax2.set_xlabel("Ciclo de operacao", fontsize=10)
    ax2.set_title("Posteriori de filtragem (Forward)", fontsize=11)
    ax2.legend(fontsize=9, ncol=2, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Experimento: {resultado['cenario']}", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparativo(resultados: list[dict], caminho: str) -> None:
    """Grade com o caminho Viterbi (cores) e as observacoes por experimento."""
    exp_plot = [r for r in resultados if "variantes" not in r
                and "posterior" in r]
    n = len(exp_plot)
    fig, ax = plt.subplots(figsize=(9, 1.1 * n + 1.5))

    for linha, r in enumerate(exp_plot):
        obs = decodificar_sequencia(r["sequencia"])
        estados_seq = [ESTADOS.index(nome) for nome in r["caminho"]]
        T = len(obs)
        for t in range(T):
            cor = CORES_ESTADO[estados_seq[t]]
            ax.add_patch(Rectangle((t, linha), 1, 1, color=cor, ec="white"))
            ax.text(t + 0.5, linha + 0.5, CODIGO_INVERSO[obs[t]],
                    ha="center", va="center", fontsize=10, color="white",
                    fontweight="bold")
        ax.text(-0.5, linha + 0.5, r["cenario"].split(". ", 1)[1],
                ha="right", va="center", fontsize=10)

    T_max = max(len(decodificar_sequencia(r["sequencia"])) for r in exp_plot)
    ax.set_xlim(-3.5, T_max + 0.5)
    ax.set_ylim(-0.3, n + 0.3)
    ax.set_yticks([])
    ax.axis("off")

    # Legenda dos estados no nivel da figura.
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in CORES_ESTADO]
    fig.legend(handles, ESTADOS, loc="lower center", ncol=4, fontsize=9,
               frameon=False, title="Estado (cor da celula)")

    fig.suptitle("Caminho Viterbi por experimento "
                 "(letra na celula = observacao n/t/v/r)", fontsize=11)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 7. MENU INTERATIVO
# ---------------------------------------------------------------------------

def menu_interativo() -> None:
    """Prompt interativo: o usuario digita sequencias de observacoes."""
    print("\n" + "=" * 70)
    print("MENU INTERATIVO — monitoramento da maquina industrial")
    print("Digite uma sequencia de observacoes (n/t/v/r) ou um comando.")
    print("Codigos: n=temp. normal  t=temp. elevada  v=vibracao  r=ruido")
    print("Comandos: (?)ajuda  (p)arametros  (e)xperimentos  (q)sair")
    print("=" * 70)

    def ajuda():
        print("\nExemplos de sequencia: 'n n n t t v v' ou 'nnnttvv'")
        print("  n = temperatura normal   t = temperatura elevada")
        print("  v = vibracao elevada     r = ruido anormal")
        print("  p = mostrar pi/A/B       e = rodar experimentos   q = sair")

    def parametros():
        print("\npi (inicial): " + ", ".join(f"{e}={PI[i]:.2f}"
              for i, e in enumerate(ESTADOS)))
        print("\nMatriz de transicao A (linhas = estado atual):")
        print("            " + " ".join(f"{e[:5]:>8}" for e in ESTADOS))
        for i, e in enumerate(ESTADOS):
            print(f"  {e:<14} " + " ".join(f"{v:8.2f}" for v in A_PADRAO[i]))
        print("\nMatriz de emissao B (linhas = estado):")
        print("            " + " ".join(f"{o[:6]:>8}" for o in OBSERVACOES))
        for i, e in enumerate(ESTADOS):
            print(f"  {e:<14} " + " ".join(f"{v:8.2f}" for v in B[i]))

    def processar(obs: list[int]):
        alfa, loglik = forward(PI, A_PADRAO, B, obs)
        caminho, logp = viterbi(PI, A_PADRAO, B, obs)
        print(f"\n  observacoes : {' '.join(CODIGO_INVERSO[o] for o in obs)}")
        print(f"  log-verossimilhanca : {loglik:.3f}")
        print(f"  caminho mais provavel (Viterbi): "
              + " -> ".join(ESTADOS[c] for c in caminho))
        for t in range(len(obs)):
            probs = ", ".join(f"{ESTADOS[i]}={alfa[t, i]:.2f}"
                              for i in range(len(ESTADOS)))
            print(f"    t={t + 1:2d} [{CODIGO_INVERSO[obs[t]]}]  {probs}")

    while True:
        try:
            entrada = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if entrada.lower() == "q":
            break
        if entrada.lower() == "?":
            ajuda()
            continue
        if entrada.lower() == "p":
            parametros()
            continue
        if entrada.lower() == "e":
            rodar_experimentos()
            continue
        try:
            obs = decodificar_sequencia(entrada)
        except ValueError as err:
            print(f"  {err}")
            continue
        if obs is None:
            print("  entrada vazia")
            continue
        processar(obs)


# ---------------------------------------------------------------------------
# 8. PERSISTENCIA E ARGUMENTOS
# ---------------------------------------------------------------------------

def salvar_resultados(resultados: list[dict]) -> None:
    """Grava JSON e CSV com os valores numericos dos experimentos."""
    caminho_json = os.path.join(DIR_RESULTADOS, "proj2_resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)

    caminho_csv = os.path.join(DIR_RESULTADOS, "proj2_tabela.csv")
    with open(caminho_csv, "w", encoding="utf-8") as fh:
        fh.write("cenario;log_verossimilhanca;log_prob_caminho;caminho\n")
        for r in resultados:
            if "variantes" in r:
                for rotulo, v in r["variantes"].items():
                    fh.write(f"{r['cenario']} [{rotulo}];"
                             f"{v['log_verossimilhanca']:.4f};"
                             f"{v['log_prob_caminho']:.4f};"
                             f"{'->'.join(v['caminho'])}\n")
            else:
                fh.write(f"{r['cenario']};{r['log_verossimilhanca']:.4f};"
                         f"{r['log_prob_caminho']:.4f};"
                         f"{'->'.join(r['caminho'])}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HMM para monitoramento de maquina industrial")
    parser.add_argument("--sequencia", type=str, default=None,
                        help="sequencia de observacoes (ex.: 'n n t t v v')")
    parser.add_argument("--nao-interativo", action="store_true",
                        help="nao abre o menu interativo")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 9. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("HMM construido (pi, A e B definidos manualmente).")

    resultados = rodar_experimentos()
    salvar_resultados(resultados)

    desenhar_modelo(os.path.join(DIR_FIGURAS, "proj2_fig1_modelo.png"))

    # Figura detalhada para o experimento de falha.
    for r in resultados:
        if r.get("cenario", "").startswith("3."):
            plot_experimento_detalhado(r, os.path.join(
                DIR_FIGURAS, "proj2_fig2_experimento_falha.png"))

    plot_comparativo(resultados, os.path.join(
        DIR_FIGURAS, "proj2_fig3_comparativo.png"))

    print("\nFiguras salvas em:", DIR_FIGURAS)
    print("Resultados salvos em:", DIR_RESULTADOS)

    if args.sequencia:
        try:
            obs = decodificar_sequencia(args.sequencia)
        except ValueError as err:
            print(f"  {err}")
            return
        alfa, loglik = forward(PI, A_PADRAO, B, obs)
        caminho, _ = viterbi(PI, A_PADRAO, B, obs)
        print(f"\nConsulta com sequencia {' '.join(CODIGO_INVERSO[o] for o in obs)}:")
        print(f"  log-verossimilhanca = {loglik:.3f}")
        print("  caminho Viterbi: " + " -> ".join(ESTADOS[c] for c in caminho))
        return

    if args.nao_interativo:
        return

    menu_interativo()


if __name__ == "__main__":
    main()
