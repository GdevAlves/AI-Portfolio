#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verificador de corrupcao textual dos arquivos da Parte 2.

Detecta dois defeitos que apareceram repetidamente durante a redacao:
  1. Caracteres de escritas nao-latina (CJK, arabe, cirilico, grego...) que
     invadiram trechos em portugues.
  2. Palavras coladas (ex.: "establisheso", "grafoconnected") por perda de
     espaco, e palavras duplicadas (ex.: "de de").

Uso:  python verificar_texto.py [arquivo ...]
"""

from __future__ import annotations

import re
import sys
import unicodedata

# Blocos nao latinos (reciclados de uma varredura Unicode).
NAO_LATINO = re.compile(
    "[\u0370-\u03FF\u0400-\u04FF\u0530-\u058F\u0590-\u05FF\u0600-\u06FF"
    "\u0700-\u074F\u0750-\u077F\u0900-\u0DFF\u0E00-\u0E7F\u0F00-\u0FFF"
    "\u1000-\u109F\u3040-\u30FF\u3130-\u318F\u3400-\u4DBF\u4E00-\u9FFF"
    "\uA960-\uA97F\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF]"
)

# "palavraMinuscula" + "PalavraMaiuscula" sem espaco no meio da frase.
# Exige que o caractere anterior NAO seja letra, para nao casar com
# identificadores camelCase legitimos (OpenStreetMap, ElementTree, HTTPError).
COLADA = re.compile(r"(?<=[\s,;:()“\"\[\]=])([a-zà-ú]{3,}[A-ZÀ-Ú][a-zà-ú]+)")

# "cobra" repetida: " de de", " que que", ...
DUPLICADA = re.compile(r"\b(que|de|da|do|para|com|uma|um|no|na)\s+\1\b", re.IGNORECASE)


def verificar(caminho: str) -> list[str]:
    with open(caminho, encoding="utf-8") as fh:
        linhas = fh.read().split("\n")

    problemas: list[str] = []
    for i, linha in enumerate(linhas, 1):
        for c in NAO_LATINO.findall(linha):
            nome = unicodedata.name(c, "?")
            problemas.append(f"linha {i}: caractere nao-latino {c!r} ({nome})")
        for m in COLADA.findall(linha):
            problemas.append(f"linha {i}: palavras coladas {m!r}")
        for m in DUPLICADA.findall(linha):
            problemas.append(f"linha {i}: palavra repetida {m!r}")
    return problemas


def main() -> int:
    arquivos = sys.argv[1:]
    if not arquivos:
        print(__doc__)
        return 2

    total = 0
    for caminho in arquivos:
        problemas = verificar(caminho)
        total += len(problemas)
        if problemas:
            print(f"\n{caminho}: {len(problemas)} problema(s)")
            for p in problemas:
                print(f"  {p}")
        else:
            print(f"{caminho}: ok")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
