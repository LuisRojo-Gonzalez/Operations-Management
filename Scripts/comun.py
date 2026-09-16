#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilidades compartidas por los cuadernos del curso.

Hace tres cosas y nada mas:

1. ENCUENTRA LOS DATOS. Los cuadernos viven en Scripts/<unidad>/ y los datos
   en DatosClases/<unidad>/. En vez de escribir "../../DatosClases/..." en cada
   celda -- que se rompe apenas alguien mueve el cuaderno o lo abre en Colab --
   se busca la carpeta subiendo por el arbol desde el directorio de trabajo.

2. GUARDA RESULTADOS de forma uniforme, en resultados/ junto al cuaderno.

3. DA UNA COMPROBACION. ``verificar(valor, esperado, "nombre")`` contrasta un
   resultado contra la cifra de referencia del curso. Sirve para que el
   estudiante sepa, sin preguntar, si su corrida reprodujo lo que debia.

Uso tipico al comienzo de un cuaderno:

    import sys; sys.path.append("..")
    from comun import datos, tabla, figura, verificar
    peso = pd.read_csv(datos("1_Calidad", "itata_peso.csv"))
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# 1. Rutas
# --------------------------------------------------------------------------
# Nombres alternativos aceptados por unidad. La carpeta de planificacion esta
# escrita "Planificiacion" en el material original; se aceptan ambas grafias
# para no depender de una errata ni obligar a renombrar carpetas del curso.
_ALIAS = {
    "1_Calidad": ["1_Calidad", "u1-calidad", "Calidad"],
    "2_Pronostico": ["2_Pronostico", "u2-pronostico", "Pronostico"],
    "3_Planificacion": ["3_Planificacion/Planificiacion",
                        "3_Planificacion/Planificacion",
                        "3_Planificacion", "u3-planificacion"],
    "3_Remanufactura": ["3_Planificacion/Remanufactura",
                        "u3-remanufactura", "Remanufactura"],
    "4_Secuenciamiento": ["4_Secuenciamiento", "u4-secuenciamiento",
                          "Secuenciamiento"],
}


def raiz_curso(desde: Path | None = None) -> Path:
    """Primera carpeta, subiendo, que contenga DatosClases/."""
    aqui = Path(desde or Path.cwd()).resolve()
    for c in [aqui, *aqui.parents]:
        if (c / "DatosClases").is_dir():
            return c
    raise FileNotFoundError(
        "No se encontro la carpeta DatosClases/ subiendo desde "
        f"{aqui}. Abra el cuaderno desde Scripts/<unidad>/ o copie los datos "
        "a una subcarpeta datos/ junto al cuaderno."
    )


def carpeta_datos(unidad: str, desde: Path | None = None) -> Path:
    """Carpeta de datos de una unidad.

    Busca, en este orden: una subcarpeta datos/ junto al cuaderno (util en
    Colab, donde se sube una sola carpeta), y luego DatosClases/<unidad>/ en
    la raiz del curso.
    """
    aqui = Path(desde or Path.cwd()).resolve()
    local = aqui / "datos"
    if local.is_dir() and any(local.glob("*.csv")):
        return local
    base = raiz_curso(aqui) / "DatosClases"
    for nombre in _ALIAS.get(unidad, [unidad]):
        c = base / nombre
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"No se encontro la carpeta de datos de '{unidad}'. Se probaron: "
        + ", ".join(str(base / n) for n in _ALIAS.get(unidad, [unidad]))
    )


def datos(unidad: str, archivo: str | None = None, desde=None) -> Path:
    """Ruta a la carpeta de datos de una unidad, o a un archivo dentro."""
    c = carpeta_datos(unidad, desde)
    if archivo is None:
        return c
    r = c / archivo
    if not r.exists():
        disponibles = sorted(p.name for p in c.glob("*.csv"))
        raise FileNotFoundError(
            f"No existe {archivo} en {c}.\nDisponibles: "
            + ", ".join(disponibles)
        )
    return r


def salida(desde: Path | None = None) -> Path:
    """Carpeta resultados/ junto al cuaderno; se crea si no existe."""
    d = Path(desde or Path.cwd()).resolve() / "resultados"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# 2. Salidas
# --------------------------------------------------------------------------
def tabla(df: pd.DataFrame, nombre: str, filas: int = 30) -> pd.DataFrame:
    """Guarda un DataFrame en resultados/<nombre>.csv y lo devuelve."""
    df.to_csv(salida() / f"{nombre}.csv", index=False)
    return df.head(filas) if len(df) > filas else df


def figura(fig, nombre: str) -> None:
    """Guarda una figura en resultados/<nombre>.png."""
    fig.tight_layout()
    fig.savefig(salida() / f"{nombre}.png", dpi=150, bbox_inches="tight")


def resumen(obj: dict, nombre: str = "resumen") -> None:
    """Guarda un diccionario de cifras en resultados/<nombre>.json."""
    def conv(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if hasattr(x, "item"):
            return x.item()
        return str(x)
    (salida() / f"{nombre}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=conv),
        encoding="utf-8")


# --------------------------------------------------------------------------
# 3. Comprobacion contra la cifra de referencia
# --------------------------------------------------------------------------
def verificar(valor, esperado, nombre: str, tol: float = 1e-3) -> bool:
    """Contrasta un resultado con la cifra de referencia del curso.

    La tolerancia es RELATIVA. Un cuaderno que no reproduce la referencia no
    esta necesariamente mal -- puede haber cambiado una semilla o una version
    del solver --, pero es una diferencia que hay que poder explicar, y para
    eso primero hay que verla.
    """
    if valor is None:
        print(f"  [  ?  ] {nombre}: sin valor")
        return False
    rel = abs(valor - esperado) / max(1.0, abs(esperado))
    ok = rel <= tol
    marca = "  OK  " if ok else " DIFIERE"
    print(f"  [{marca}] {nombre}: obtenido {valor:,.4f} | "
          f"referencia {esperado:,.4f} | dif. relativa {rel:.2e}")
    return ok


# --------------------------------------------------------------------------
# 4. Estilo de graficos
# --------------------------------------------------------------------------
def estilo() -> None:
    """Estilo sobrio y legible al proyectar en sala."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (10, 4.2),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "font.size": 10,
        "legend.frameon": False,
    })


__all__ = ["raiz_curso", "carpeta_datos", "datos", "salida", "tabla",
           "figura", "resumen", "verificar", "estilo"]
