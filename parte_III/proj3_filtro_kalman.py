#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO 3 - FILTRO DE KALMAN
=============================
Deteccao de obstaculos com sensor de distancia (HC-SR04 / ultrassonico) e
tomada de decisao robusta a ruido, usando um Filtro de Kalman unidimensional.

MOTIVACAO
  Um robo mede continuamente a distancia ate um obstaculo. O sensor entrega
  medicoes ruidosas (ex.: 49, 52, 47, 55, 48 cm para uma distancia real de
  50 cm). Se o robo decidir "continuar / reduzir / parar" diretamente sobre
  essas medicoes, podera reagir a picos de ruido e tomar decisoes erradas.
  O Filtro de Kalman combina a PREDICAO de um modelo de movimento com cada
  NOVA MEDICAO, produzindo uma estimativa mais estavel e confiavel.

MODELO (linear, velocidade constante)
  Vetor de estado:      X = [d, v]      (distancia e taxa de variacao)
  Transicao (F):        d_k = d_{k-1} + v_{k-1}*dt ; v_k = v_{k-1}
  Observacao (H):       z_k = d_k + ruido  (mede apenas a distancia)
  Ruido de processo Q:  incerteza do modelo (a velocidade pode variar)
  Ruido de medicao R:   variancia do ruido do sensor

REGRAS DE DECISAO (limiares experimentais)
  d > 50 cm  -> Continuar
  20 <= d <= 50 cm -> Reduzir velocidade
  d < 20 cm  -> Parar

ALGORITMO (implementado manualmente, sem bibliotecas de filtragem)
  Predicao    : x_barra = F x        ;  P_barra = F P F^T + Q
  Atualizacao : K = P_barra H^T (H P_barra H^T + R)^-1   (ganho de Kalman)
                x = x_barra + K (z - H x_barra)
                P = (I - K H) P_barra

EXECUCAO
  python proj3_filtro_kalman.py
  Roda os 4 experimentos (gerando figuras e valores para o relatorio).
  Flags opcionais: --experimento N, --ruido VALOR, --nao-interativo.

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

# ===========================================================================
# 1. CONFIGURACAO
# ===========================================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURAS = os.path.join(BASE, "figures")
DIR_RESULTADOS = os.path.join(BASE, "results")

os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Limiares de decisao (em cm).
LIMIAR_CONTINUAR = 50.0
LIMIAR_PARAR = 20.0

# ---------------------------------------------------------------------------
# 2. SIMULACAO DO SENSOR (distancia real + ruido)
# ---------------------------------------------------------------------------

def gerar_real(cenario: str, N: int) -> np.ndarray:
    """Devolve a sequencia de distancias reais (sem ruido) para um cenario."""
    k = np.arange(N)
    if cenario == "aproximacao":
        # Obstaculo se aproxima linearmente de 100 cm a 10 cm.
        return np.linspace(100.0, 10.0, N)
    if cenario == "picos":
        # Distancia constante em 35 cm (zona "Reduzir").
        return np.full(N, 35.0)
    # "pouco_ruido" e "muito_ruido": distancia constante em 40 cm, no centro
    # da zona "Reduzir" (longe dos limiares 20 e 50), para que o ruido cause
    # decisoes erradas ocasionais e o filtro as corrija.
    return np.full(N, 40.0)


def gerar_medicoes(real: np.ndarray, sigma: float,
                   picos: dict[int, float] | None = None,
                   semente: int = 42) -> np.ndarray:
    """Adiciona ruido gaussiano N(0, sigma^2) e, opcionalmente, picos/outliers."""
    rng = np.random.default_rng(semente)
    medicoes = real + rng.normal(0.0, sigma, size=real.shape)
    if picos:
        for k, valor in picos.items():
            medicoes[k] = real[k] + valor
    return medicoes


# ---------------------------------------------------------------------------
# 3. FILTRO DE KALMAN
# ---------------------------------------------------------------------------

def construir_filtro(sigma: float) -> dict:
    """Monta as matrizes F, H, Q, R, o estado inicial x0 e a covariancia P0."""
    dt = 1.0
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    # Ruido de processo: pouca incerteza na distancia, mais na velocidade
    # (a velocidade pode variar entre amostras; e o "erro do modelo").
    Q = np.array([[0.01, 0.0],
                  [0.0, 0.10]])
    # Ruido de medicao: variancia do ruido do sensor.
    R = np.array([[sigma ** 2]])
    # Covariancia inicial: incerteza da distancia = ruido do sensor; a
    # velocidade e desconhecida, entao recebe incerteza grande.
    P0 = np.array([[sigma ** 2, 0.0],
                   [0.0, 10.0]])
    return {"F": F, "H": H, "Q": Q, "R": R, "P0": P0, "dt": dt}


def predicao(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray):
    """Etapa de predicao: projeta o estado e a covariancia um passo adiante."""
    x_barra = F @ x
    P_barra = F @ P @ F.T + Q
    return x_barra, P_barra


def atualizacao(x_barra: np.ndarray, P_barra: np.ndarray, z: float,
                H: np.ndarray, R: np.ndarray):
    """Etapa de atualizacao: corrige a predicao usando a nova medicao z."""
    S = H @ P_barra @ H.T + R          # inovacao (covariancia do residuo)
    K = P_barra @ H.T @ np.linalg.inv(S)   # ganho de Kalman
    x = x_barra + K @ (z - H @ x_barra)
    P = (np.eye(len(x_barra)) - K @ H) @ P_barra
    return x, P


def rodar_filtro(medicoes: np.ndarray, filtro: dict):
    """Processa todas as medicoes e devolve estados estimados + variancias.

    O estado inicial parte da primeira medicao com velocidade zero.
    """
    F, H, Q, R, P0 = filtro["F"], filtro["H"], filtro["Q"], filtro["R"], filtro["P0"]
    N = len(medicoes)
    x = np.array([medicoes[0], 0.0])
    P = P0.copy()

    estimativas = np.zeros((N, 2))
    variancias = np.zeros(N)   # variancia da distancia (P[0,0]) para a banda 2sigma
    estimativas[0] = x
    variancias[0] = P[0, 0]

    for k in range(1, N):
        x, P = predicao(x, P, F, Q)
        x, P = atualizacao(x, P, medicoes[k], H, R)
        estimativas[k] = x
        variancias[k] = P[0, 0]

    return estimativas, variancias


# ---------------------------------------------------------------------------
# 4. DECISAO E AVALIACAO
# ---------------------------------------------------------------------------

def decidir(d: float) -> str:
    """Aplica os limiares para transformar distancia em acao."""
    if d > LIMIAR_CONTINUAR:
        return "Continuar"
    if d < LIMIAR_PARAR:
        return "Parar"
    return "Reduzir"


def avaliar(real: np.ndarray, medicoes: np.ndarray,
            estimativas: np.ndarray) -> dict:
    """Calcula RMSE/MAE e o numero de decisoes incorretas (medidas vs filtro)."""
    decisao_correta = [decidir(d) for d in real]
    decisao_medida = [decidir(d) for d in medicoes]
    decisao_filtro = [decidir(d) for d in estimativas[:, 0]]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    incorretas_medida = sum(1 for c, m in zip(decisao_correta, decisao_medida)
                            if c != m)
    incorretas_filtro = sum(1 for c, m in zip(decisao_correta, decisao_filtro)
                            if c != m)

    return {
        "rmse_medida": rmse(medicoes, real),
        "rmse_filtro": rmse(estimativas[:, 0], real),
        "mae_medida": mae(medicoes, real),
        "mae_filtro": mae(estimativas[:, 0], real),
        "decisoes_incorretas_medida": incorretas_medida,
        "decisoes_incorretas_filtro": incorretas_filtro,
    }


# ---------------------------------------------------------------------------
# 5. EXPERIMENTOS
# ---------------------------------------------------------------------------

EXPERIMENTOS = [
    {"id": 1, "nome": "Pouco ruido", "cenario": "pouco_ruido", "sigma": 1.0},
    {"id": 2, "nome": "Muito ruido", "cenario": "muito_ruido", "sigma": 8.0},
    {"id": 3, "nome": "Aproximacao", "cenario": "aproximacao", "sigma": 3.0},
    {"id": 4, "nome": "Picos", "cenario": "picos", "sigma": 2.0,
     "picos": {10: 30.0, 20: -28.0, 30: 35.0, 40: -26.0}},
]

N_AMOSTRAS = 50


def rodar_experimentos() -> list[dict]:
    """Executa todos os experimentos e devolve os resultados estruturados."""
    resultados = []
    print("=" * 70)
    print("RESULTADOS DOS EXPERIMENTOS")
    print("=" * 70)

    for exp in EXPERIMENTOS:
        real = gerar_real(exp["cenario"], N_AMOSTRAS)
        medicoes = gerar_medicoes(real, exp["sigma"], exp.get("picos"))
        filtro = construir_filtro(exp["sigma"])
        estimativas, variancias = rodar_filtro(medicoes, filtro)
        metricas = avaliar(real, medicoes, estimativas)

        resultado = {
            "experimento": exp["nome"],
            "sigma": exp["sigma"],
            "real": real.tolist(),
            "medicoes": medicoes.tolist(),
            "estimativa": estimativas[:, 0].tolist(),
            "variancia": variancias.tolist(),
            **metricas,
        }
        resultados.append(resultado)

        print(f"\n{exp['id']}. {exp['nome']} (sigma={exp['sigma']})")
        print(f"    RMSE medidas={metricas['rmse_medida']:.3f} cm   "
              f"RMSE filtro={metricas['rmse_filtro']:.3f} cm")
        print(f"    Decisoes incorretas: medidas={metricas['decisoes_incorretas_medida']}"
              f"  filtro={metricas['decisoes_incorretas_filtro']}")

    return resultados


# ---------------------------------------------------------------------------
# 6. VISUALIZACAO
# ---------------------------------------------------------------------------

def plot_aproximacao(res: dict, caminho: str) -> None:
    """Distancia ao longo do tempo (aproximacao) com banda 2sigma e limiares."""
    k = np.arange(len(res["real"]))
    real = np.array(res["real"])
    med = np.array(res["medicoes"])
    est = np.array(res["estimativa"])
    sigma = np.sqrt(np.array(res["variancia"]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k, real, color="black", lw=1.8, label="Distancia real")
    ax.plot(k, med, "o", color="#7f7f7f", ms=4, alpha=0.6, label="Medicoes")
    ax.plot(k, est, color="#d7191c", lw=1.8, label="Estimativa (Kalman)")
    ax.fill_between(k, est - 2 * sigma, est + 2 * sigma,
                    color="#d7191c", alpha=0.15, label="Intervalo $\\pm 2\\sigma$")

    ax.axhline(LIMIAR_CONTINUAR, color="green", ls="--", lw=1.2,
               label="Limiar Continuar (50 cm)")
    ax.axhline(LIMIAR_PARAR, color="blue", ls="--", lw=1.2,
               label="Limiar Parar (20 cm)")

    ax.set_xlabel("Amostra (tempo)", fontsize=11)
    ax.set_ylabel("Distancia (cm)", fontsize=11)
    ax.set_title("Aproximacao do obstaculo (Experimento 3)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def plot_picos(res: dict, caminho: str) -> None:
    """Experimento com picos: destaca outliers e decisoes erradas/corretas."""
    k = np.arange(len(res["real"]))
    real = np.array(res["real"])
    med = np.array(res["medicoes"])
    est = np.array(res["estimativa"])

    # Identifica os picos (desvios grandes da media local).
    picos_idx = np.where(np.abs(med - real) > 5.0)[0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k, real, color="black", lw=1.8, label="Distancia real (35 cm)")
    ax.plot(k, med, "o", color="#7f7f7f", ms=4, alpha=0.5, label="Medicoes")
    ax.plot(k, est, color="#d7191c", lw=1.8, label="Estimativa (Kalman)")
    ax.plot(picos_idx, med[picos_idx], "o", color="#ff7f00", ms=8,
            label="Picos (outliers)")

    ax.axhline(LIMIAR_CONTINUAR, color="green", ls="--", lw=1.2)
    ax.axhline(LIMIAR_PARAR, color="blue", ls="--", lw=1.2)
    ax.text(len(k) - 1, LIMIAR_CONTINUAR + 2, "Continuar", color="green", ha="right", fontsize=9)
    ax.text(len(k) - 1, (LIMIAR_CONTINUAR + LIMIAR_PARAR) / 2, "Reduzir",
            color="darkgoldenrod", ha="right", fontsize=9)
    ax.text(len(k) - 1, LIMIAR_PARAR - 3, "Parar", color="blue", ha="right", fontsize=9)

    ax.set_xlabel("Amostra (tempo)", fontsize=11)
    ax.set_ylabel("Distancia (cm)", fontsize=11)
    ax.set_title("Medicoes com picos de ruido (Experimento 4)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def plot_resumo(resultados: list[dict], caminho: str) -> None:
    """Barras: RMSE (medidas vs filtro) e decisoes incorretas por experimento."""
    nomes = [r["experimento"] for r in resultados]
    x = np.arange(len(nomes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    largura = 0.38
    ax1.bar(x - largura / 2, [r["rmse_medida"] for r in resultados],
            largura, label="Medicoes", color="#7f7f7f")
    ax1.bar(x + largura / 2, [r["rmse_filtro"] for r in resultados],
            largura, label="Filtro", color="#d7191c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes, fontsize=9)
    ax1.set_ylabel("RMSE (cm)", fontsize=10)
    ax1.set_title("Erro de estimacao (RMSE)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x - largura / 2, [r["decisoes_incorretas_medida"] for r in resultados],
            largura, label="Medicoes", color="#7f7f7f")
    ax2.bar(x + largura / 2, [r["decisoes_incorretas_filtro"] for r in resultados],
            largura, label="Filtro", color="#d7191c")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nomes, fontsize=9)
    ax2.set_ylabel("Decisoes incorretas", fontsize=10)
    ax2.set_title("Robustez da decisao", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 7. PERSISTENCIA E ARGUMENTOS
# ---------------------------------------------------------------------------

def salvar_resultados(resultados: list[dict]) -> None:
    """Grava JSON e CSV com os valores numericos."""
    caminho_json = os.path.join(DIR_RESULTADOS, "proj3_resultados.json")
    resumo = [{k: v for k, v in r.items()
               if k not in ("real", "medicoes", "estimativa", "variancia")}
              for r in resultados]
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2)

    caminho_csv = os.path.join(DIR_RESULTADOS, "proj3_tabela.csv")
    with open(caminho_csv, "w", encoding="utf-8") as fh:
        fh.write("experimento;sigma;rmse_medida;rmse_filtro;mae_medida;"
                 "mae_filtro;dec_incorretas_medida;dec_incorretas_filtro\n")
        for r in resumo:
            fh.write(f"{r['experimento']};{r['sigma']};{r['rmse_medida']:.4f};"
                     f"{r['rmse_filtro']:.4f};{r['mae_medida']:.4f};"
                     f"{r['mae_filtro']:.4f};{r['decisoes_incorretas_medida']};"
                     f"{r['decisoes_incorretas_filtro']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtro de Kalman para deteccao de obstaculos")
    parser.add_argument("--experimento", type=int, default=None,
                        help="roda apenas um experimento (1 a 4)")
    parser.add_argument("--ruido", type=float, default=None,
                        help="desvio padrao do ruido (sobrescreve o default)")
    parser.add_argument("--nao-interativo", action="store_true",
                        help="nao mostra saida detalhada")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 8. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    experimentos = EXPERIMENTOS
    if args.experimento is not None:
        experimentos = [e for e in EXPERIMENTOS if e["id"] == args.experimento]
        if not experimentos:
            print("experimento invalido (use 1 a 4)")
            return
    if args.ruido is not None:
        for e in experimentos:
            e["sigma"] = args.ruido

    resultados = rodar_experimentos()
    salvar_resultados(resultados)

    por_nome = {r["experimento"]: r for r in resultados}
    plot_aproximacao(por_nome["Aproximacao"],
                     os.path.join(DIR_FIGURAS, "proj3_fig1_aproximacao.png"))
    plot_picos(por_nome["Picos"],
               os.path.join(DIR_FIGURAS, "proj3_fig2_picos.png"))
    plot_resumo(resultados,
                os.path.join(DIR_FIGURAS, "proj3_fig3_resumo.png"))

    print("\nFiguras salvas em:", DIR_FIGURAS)
    print("Resultados salvos em:", DIR_RESULTADOS)


if __name__ == "__main__":
    main()
