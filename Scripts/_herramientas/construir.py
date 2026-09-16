#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convierte los fuentes .py anotados en cuadernos .ipynb y los ejecuta.

Formato del fuente: celdas separadas por lineas marcadoras.

    # %% [markdown]
    # Texto en markdown, una almohadilla por linea.

    # %%
    codigo de python

Uso:
    python _herramientas/construir.py                 # todo
    python _herramientas/construir.py 3_Planificacion # una unidad
    python _herramientas/construir.py --sin-ejecutar
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
FUENTES = RAIZ / "_fuentes"


def partir(texto: str):
    """Devuelve [(tipo, contenido), ...] a partir del fuente anotado."""
    celdas, tipo, buf = [], "code", []

    def cerrar():
        cuerpo = "\n".join(buf).strip("\n")
        if cuerpo.strip():
            celdas.append((tipo, cuerpo))

    for linea in texto.splitlines():
        if linea.startswith("# %% [markdown]"):
            cerrar()
            tipo, buf = "markdown", []
        elif linea.startswith("# %%"):
            cerrar()
            tipo, buf = "code", []
        else:
            buf.append(linea)
    cerrar()

    out = []
    for tipo, cuerpo in celdas:
        if tipo == "markdown":
            # quitar el "# " inicial de cada linea
            cuerpo = "\n".join(
                l[2:] if l.startswith("# ") else (l[1:] if l == "#" else l)
                for l in cuerpo.splitlines())
        out.append((tipo, cuerpo.strip("\n")))
    return out


def tiene_salidas(ruta: Path) -> bool:
    """True si el cuaderno ya fue ejecutado y conserva sus salidas."""
    if not ruta.exists():
        return False
    try:
        nb = nbformat.read(ruta, as_version=4)
    except Exception:
        return False
    return any(c.get("outputs") for c in nb.cells if c.cell_type == "code")


def construir(fuente: Path, destino: Path, forzar: bool = False) -> Path | None:
    """Escribe el .ipynb desde el fuente.

    Un cuaderno YA EJECUTADO cuyo fuente no cambió NO se toca. Reconstruirlo
    lo dejaria sin salidas, que es justo lo contrario de lo que se quiere en
    una guia de estudio: las salidas son el contenido. Solo se regenera si el
    fuente es mas nuevo que el cuaderno, o si se pide --forzar.
    """
    if not forzar and tiene_salidas(destino):
        if destino.stat().st_mtime >= fuente.stat().st_mtime:
            return None

    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    for tipo, cuerpo in partir(fuente.read_text(encoding="utf-8")):
        nb.cells.append(nbformat.v4.new_markdown_cell(cuerpo) if tipo == "markdown"
                        else nbformat.v4.new_code_cell(cuerpo))
    destino.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destino)
    return destino


def ejecutar(ruta: Path, timeout: int = 1200) -> tuple[bool, str]:
    nb = nbformat.read(ruta, as_version=4)
    cli = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                         resources={"metadata": {"path": str(ruta.parent)}},
                         allow_errors=False)
    try:
        cli.execute()
        nbformat.write(nb, ruta)
        return True, ""
    except Exception as e:
        nbformat.write(nb, ruta)
        return False, f"{type(e).__name__}: {str(e)[:600]}"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    correr = "--sin-ejecutar" not in sys.argv
    forzar = "--forzar" in sys.argv
    filtro = args[0] if args else None

    fuentes = sorted(FUENTES.rglob("*.py"))
    if filtro:
        fuentes = [f for f in fuentes if filtro in str(f)]
    if not fuentes:
        print(f"Sin fuentes que coincidan con {filtro!r} en {FUENTES}")
        return 1

    fallos = []
    for f in fuentes:
        rel = f.relative_to(FUENTES)
        destino = RAIZ / rel.with_suffix(".ipynb")
        hecho = construir(f, destino, forzar=forzar)
        if hecho is None:
            print(f"  sin cambios {str(rel.with_suffix('.ipynb')):<52} "
                  f"(ya ejecutado; --forzar para regenerar)")
            continue
        if not correr:
            print(f"  construido  {str(rel.with_suffix('.ipynb')):<52} SIN salidas")
            continue
        t0 = time.perf_counter()
        ok, err = ejecutar(destino)
        seg = time.perf_counter() - t0
        print(f"  {'OK   ' if ok else 'FALLA'}  {str(rel.with_suffix('.ipynb')):<52} {seg:6.1f}s")
        if not ok:
            print(f"         {err}")
            fallos.append((rel, err))

    print()
    print(f"{len(fuentes) - len(fallos)}/{len(fuentes)} cuadernos ejecutados sin error")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
