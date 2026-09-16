#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Datos DIDACTICOS para la Unidad de Calidad -- Conservas del Itata S.A.

Planta PARALELA al caso integrador (Bebidas del Maipo). Deliberadamente
distinta en producto, caracteristica medida, tamanio de subgrupo y
patologias, para que los estudiantes aprendan el metodo en clase sin recibir
resueltos los hallazgos de su Entrega 1.

  BdM (caso)            ->  Itata (clase)
  volumen de llenado    ->  peso escurrido
  n = 5                 ->  n = 4     (obliga a buscar otras constantes)
  turno noche disperso  ->  corrimiento de media en Fase II
  lote de etiquetas     ->  falla de sellado
  (sin MSA)             ->  estudio R&R marginal

Genera:
  itata_peso.csv        30 subgrupos x n=4  (Fase I: 1-20, Fase II: 21-30)
  itata_atributos.csv   24 lotes con n VARIABLE (limites de carta p variables)
  itata_msa.csv         10 piezas x 3 operadores x 2 repeticiones
  oc_plan.csv           curva OC del plan de aceptacion n=80, c=2

Uso:  python generar_datos_clase.py [--seed 424242]
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------- parametros

NOMINAL = 480.0          # peso escurrido declarado en el envase, g
LIE, LSE = 465.0, 495.0  # especificacion de ingenieria
TOLERANCIA = LSE - LIE

MU_FASE1 = 486.0         # el proceso corre DESCENTRADO hacia arriba (sobrellenado defensivo)
SIGMA_TOT = 4.10         # desviacion observada = proceso + medicion
N_SUB = 4                # tamanio de subgrupo
N_FASE1 = 20
N_FASE2 = 10
SUBGRUPO_CORRIMIENTO = 26
MU_CORRIMIENTO = 492.0

# Estudio R&R: buscamos deliberadamente la zona MARGINAL
SIGMA_REPETIBILIDAD = 1.00
BIAS_OPERADOR = {"A": 0.0, "B": 0.95, "C": -0.60}
SIGMA_PIEZA_ESTUDIO = 3.90

# Atributos
P_BARRA = 0.038
LOTE_MALO = 15
P_LOTE_MALO = 0.098
CAUSAS = ["Sello defectuoso", "Rotulado", "Peso fuera de rango",
          "Abolladura", "Cuerpo extranio"]
PESOS_CAUSA = np.array([0.30, 0.27, 0.21, 0.14, 0.08])
PESOS_CAUSA_MALO = np.array([0.78, 0.08, 0.06, 0.05, 0.03])

# Plan de muestreo de aceptacion
LOTE_N = 2000
PLAN_n, PLAN_c = 80, 2

# Constantes de cartas de control (AIAG / Montgomery)
CONST = {
    2: dict(A2=1.880, D3=0.000, D4=3.267, d2=1.128, A3=2.659, B3=0.000, B4=3.267, c4=0.7979),
    3: dict(A2=1.023, D3=0.000, D4=2.574, d2=1.693, A3=1.954, B3=0.000, B4=2.568, c4=0.8862),
    4: dict(A2=0.729, D3=0.000, D4=2.282, d2=2.059, A3=1.628, B3=0.000, B4=2.266, c4=0.9213),
    5: dict(A2=0.577, D3=0.000, D4=2.114, d2=2.326, A3=1.427, B3=0.000, B4=2.089, c4=0.9400),
}

# Constantes del metodo de rangos para R&R (AIAG MSA 4a ed.)
K1_2TRIALS = 0.8862      # 2 repeticiones
K2_3OPER = 0.5231        # 3 operadores
K3_10PIEZAS = 0.3146     # 10 piezas


# --------------------------------------------------------------- generacion

def generar_peso(rng):
    filas = []
    for sg in range(1, N_FASE1 + N_FASE2 + 1):
        fase = "I" if sg <= N_FASE1 else "II"
        mu = MU_CORRIMIENTO if sg >= SUBGRUPO_CORRIMIENTO else MU_FASE1
        turno = ["A (06-14)", "B (14-22)"][(sg - 1) % 2]
        obs = rng.normal(mu, SIGMA_TOT, N_SUB)
        for j, v in enumerate(obs, start=1):
            filas.append(dict(subgrupo=sg, fase=fase, turno=turno,
                              n_obs=j, peso_g=round(float(v), 2)))
    return pd.DataFrame(filas)


def generar_atributos(rng):
    filas = []
    for lote in range(1, 25):
        n = int(rng.integers(250, 401))
        p = P_LOTE_MALO if lote == LOTE_MALO else P_BARRA
        pesos = PESOS_CAUSA_MALO if lote == LOTE_MALO else PESOS_CAUSA
        x = int(rng.binomial(n, p))
        causa = CAUSAS[int(rng.choice(len(CAUSAS), p=pesos))] if x else "Sin defectos"
        filas.append(dict(lote=lote, n_inspeccionadas=n, n_defectuosas=x,
                          causa_principal=causa))
    return pd.DataFrame(filas)


def generar_msa(rng):
    piezas = rng.normal(NOMINAL + 4, SIGMA_PIEZA_ESTUDIO, 10)
    filas = []
    for i, valor in enumerate(piezas, start=1):
        for op, bias in BIAS_OPERADOR.items():
            for rep in (1, 2):
                m = valor + bias + rng.normal(0, SIGMA_REPETIBILIDAD)
                filas.append(dict(pieza=i, operador=op, repeticion=rep,
                                  medicion_g=round(float(m), 2)))
    return pd.DataFrame(filas)


def curva_oc(N=LOTE_N, n=PLAN_n, c=PLAN_c):
    """Probabilidad de aceptacion bajo binomial e hipergeometrica."""
    filas = []
    for p in np.arange(0.0, 0.1001, 0.0025):
        pa_bin = stats.binom.cdf(c, n, p)
        D = int(round(p * N))
        pa_hyp = stats.hypergeom.cdf(c, N, D, n)
        filas.append(dict(p=round(float(p), 4),
                          Pa_binomial=round(float(pa_bin), 4),
                          Pa_hipergeometrica=round(float(pa_hyp), 4)))
    return pd.DataFrame(filas)


# --------------------------------------------------------------- analisis

def analizar(peso, atr, msa, oc):
    """Imprime TODAS las cifras que se citan en las laminas."""
    n = N_SUB
    K = CONST[n]
    sep = "=" * 74

    print(sep); print("A. CARTA X-BARRA / R  (Conservas del Itata)"); print(sep)
    f1 = peso[peso.fase == "I"]
    sg = f1.groupby("subgrupo")["peso_g"]
    xbar, R = sg.mean(), sg.max() - sg.min()
    xbb, Rb = xbar.mean(), R.mean()
    print(f"\nFase I: {len(xbar)} subgrupos de n={n}")
    print(f"  X-doble-barra = {xbb:.3f} g      R-barra = {Rb:.3f} g")
    print(f"  Constantes n={n}: A2={K['A2']}, D3={K['D3']}, D4={K['D4']}, d2={K['d2']}")
    lcs_x, lci_x = xbb + K["A2"] * Rb, xbb - K["A2"] * Rb
    lcs_r, lci_r = K["D4"] * Rb, K["D3"] * Rb
    print(f"\n  Carta X-barra: LC={xbb:.2f}  LCS={lcs_x:.2f}  LCI={lci_x:.2f}")
    print(f"  Carta R:       LC={Rb:.2f}  LCS={lcs_r:.2f}  LCI={lci_r:.2f}")
    sigma = Rb / K["d2"]
    print(f"\n  sigma_hat = R-barra/d2 = {Rb:.3f}/{K['d2']} = {sigma:.3f} g")
    fuera1 = xbar[(xbar > lcs_x) | (xbar < lci_x)]
    print(f"  Subgrupos fuera de control en Fase I: "
          f"{list(fuera1.index) if len(fuera1) else 'ninguno (proceso estable)'}")

    print(f"\n  --- Fase II (limites congelados de Fase I) ---")
    f2 = peso[peso.fase == "II"]
    sg2 = f2.groupby("subgrupo")["peso_g"]
    xbar2 = sg2.mean()
    for s, v in xbar2.items():
        flag = ""
        if v > lcs_x: flag = "  <== FUERA (sobre LCS)"
        elif v < lci_x: flag = "  <== FUERA (bajo LCI)"
        print(f"    subgrupo {s}: X-barra = {v:7.2f}{flag}")
    # racha sobre la linea central
    racha, mejor = 0, 0
    for v in xbar2:
        racha = racha + 1 if v > xbb else 0
        mejor = max(mejor, racha)
    print(f"  Racha maxima consecutiva sobre LC en Fase II: {mejor} puntos")

    print(); print(sep); print("B. CAPACIDAD Y DESEMPENIO"); print(sep)
    Cp = TOLERANCIA / (6 * sigma)
    Cpu, Cpl = (LSE - xbb) / (3 * sigma), (xbb - LIE) / (3 * sigma)
    Cpk = min(Cpu, Cpl)
    todos = f1["peso_g"]
    s_lp = todos.std(ddof=1)
    Pp = TOLERANCIA / (6 * s_lp)
    Ppk = min((LSE - todos.mean()) / (3 * s_lp), (todos.mean() - LIE) / (3 * s_lp))
    k = abs(xbb - (LIE + LSE) / 2) / (TOLERANCIA / 2)
    print(f"\n  Especificacion: {NOMINAL:.0f} g  [LIE {LIE:.0f} ; LSE {LSE:.0f}]"
          f"   Tolerancia = {TOLERANCIA:.0f} g")
    print(f"  sigma corto plazo (R-barra/d2) = {sigma:.3f} g")
    print(f"  sigma largo plazo (s muestral) = {s_lp:.3f} g")
    print(f"\n  Cp  = {Cp:.3f}      Cpu = {Cpu:.3f}   Cpl = {Cpl:.3f}   Cpk = {Cpk:.3f}")
    print(f"  Pp  = {Pp:.3f}      Ppk = {Ppk:.3f}")
    print(f"  k (indice de descentrado) = {k:.3f}   -> Cpk = Cp(1-k) = {Cp*(1-k):.3f}")
    print(f"  Cp/Cpk = {Cp/Cpk:.2f}  (si fuera 1, el proceso estaria centrado)")
    ppm_sup = stats.norm.sf((LSE - xbb) / sigma) * 1e6
    ppm_inf = stats.norm.cdf((LIE - xbb) / sigma) * 1e6
    print(f"\n  PPM esperado sobre LSE = {ppm_sup:,.0f}   bajo LIE = {ppm_inf:,.0f}"
          f"   total = {ppm_sup + ppm_inf:,.0f}")
    print(f"  Si se RECENTRA en {(LIE+LSE)/2:.0f} g con el mismo sigma:")
    ppm_c = 2 * stats.norm.sf((TOLERANCIA / 2) / sigma) * 1e6
    print(f"     Cpk pasaria de {Cpk:.2f} a {Cp:.2f}  y el PPM de "
          f"{ppm_sup+ppm_inf:,.0f} a {ppm_c:,.0f}")
    print(f"  Sobrepeso medio regalado = {xbb - NOMINAL:+.2f} g por envase")

    print(); print(sep); print("C. ESTUDIO R&R (metodo de rangos, AIAG)"); print(sep)
    piv = msa.pivot_table(index=["pieza", "operador"], columns="repeticion",
                          values="medicion_g")
    piv["R"] = (piv[1] - piv[2]).abs()
    Rbb = piv["R"].mean()
    EV = Rbb * K1_2TRIALS
    med_op = msa.groupby("operador")["medicion_g"].mean()
    Xdiff = med_op.max() - med_op.min()
    n_piezas, n_rep = 10, 2
    AV2 = (Xdiff * K2_3OPER) ** 2 - EV ** 2 / (n_piezas * n_rep)
    AV = np.sqrt(max(AV2, 0.0))
    GRR = np.sqrt(EV ** 2 + AV ** 2)
    med_pieza = msa.groupby("pieza")["medicion_g"].mean()
    Rp = med_pieza.max() - med_pieza.min()
    PV = Rp * K3_10PIEZAS
    TV = np.sqrt(GRR ** 2 + PV ** 2)
    ndc = 1.41 * PV / GRR
    print(f"\n  Medias por operador: " +
          "  ".join(f"{o}={v:.2f}" for o, v in med_op.items()))
    print(f"  R-doble-barra = {Rbb:.3f} g     X-diff = {Xdiff:.3f} g"
          f"     R-piezas = {Rp:.3f} g")
    print(f"\n  EV  (repetibilidad, equipo)     = {EV:.3f} g")
    print(f"  AV  (reproducibilidad, operador)= {AV:.3f} g")
    print(f"  GRR = sqrt(EV^2+AV^2)           = {GRR:.3f} g")
    print(f"  PV  (variacion entre piezas)    = {PV:.3f} g")
    print(f"  TV  (variacion total)           = {TV:.3f} g")
    print(f"\n  %GRR sobre variacion total   = {100*GRR/TV:.1f} %")
    print(f"  %GRR sobre tolerancia (6*GRR/Tol) = {100*6*GRR/TOLERANCIA:.1f} %")
    print(f"  ndc = 1.41*PV/GRR = {ndc:.2f}  -> {int(ndc)} categorias distintas")
    print(f"  Criterio AIAG: <10% aceptable | 10-30% marginal | >30% inaceptable;"
          f" ndc >= 5")
    sigma_proc = np.sqrt(max(sigma ** 2 - GRR ** 2, 0.0))
    print(f"\n  Descomposicion:  sigma_obs^2 = sigma_proceso^2 + sigma_medicion^2")
    print(f"     {sigma:.3f}^2 = sigma_proc^2 + {GRR:.3f}^2"
          f"  ->  sigma_proceso = {sigma_proc:.3f} g")
    print(f"     Cp OBSERVADO   = {TOLERANCIA/(6*sigma):.3f}")
    print(f"     Cp VERDADERO   = {TOLERANCIA/(6*sigma_proc):.3f}"
          f"   (el instrumento se come {TOLERANCIA/(6*sigma_proc)-Cp:.3f} de Cp)")

    print(); print(sep); print("D. CARTA p  (n variable)"); print(sep)
    pb = atr.n_defectuosas.sum() / atr.n_inspeccionadas.sum()
    print(f"\n  p-barra = {atr.n_defectuosas.sum()}/{atr.n_inspeccionadas.sum()}"
          f" = {pb:.4f}")
    print(f"  n varia entre {atr.n_inspeccionadas.min()} y "
          f"{atr.n_inspeccionadas.max()} -> limites VARIABLES\n")
    fuera = []
    for _, r in atr.iterrows():
        lcs = pb + 3 * np.sqrt(pb * (1 - pb) / r.n_inspeccionadas)
        lci = max(0.0, pb - 3 * np.sqrt(pb * (1 - pb) / r.n_inspeccionadas))
        p_i = r.n_defectuosas / r.n_inspeccionadas
        if p_i > lcs or p_i < lci:
            fuera.append((int(r.lote), p_i, lcs, r.causa_principal))
    for lote, p_i, lcs, causa in fuera:
        print(f"  Lote {lote}: p={p_i:.4f} > LCS={lcs:.4f}  causa: {causa}"
              f"   <== FUERA DE CONTROL")
    if not fuera:
        print("  (ningun lote fuera de control)")
    n_med = atr.n_inspeccionadas.mean()
    print(f"\n  Con n medio = {n_med:.0f}: LCS = "
          f"{pb + 3*np.sqrt(pb*(1-pb)/n_med):.4f}")
    print("\n  Pareto de causas (unidades defectuosas):")
    par = atr.groupby("causa_principal")["n_defectuosas"].sum().sort_values(ascending=False)
    acum = 0
    for c, v in par.items():
        acum += v
        print(f"    {c:<22} {v:>4}  acumulado {100*acum/par.sum():5.1f} %")

    print(); print(sep); print("E. MUESTREO DE ACEPTACION"); print(sep)
    print(f"\n  Lote N={LOTE_N}, plan simple n={PLAN_n}, c={PLAN_c}")
    for p_ in (0.01, 0.02, 0.03, 0.05):
        pa = stats.binom.cdf(PLAN_c, PLAN_n, p_)
        print(f"    p={p_:.0%}  ->  Pa = {pa:.4f}   (riesgo de rechazo {1-pa:.4f})")
    pa_aql = stats.binom.cdf(PLAN_c, PLAN_n, 0.01)
    print(f"\n  AQL = 1 %   -> riesgo del productor  alpha = {1-pa_aql:.4f}")
    # LTPD al 10 %
    ps = np.arange(0.001, 0.15, 0.0001)
    pas = stats.binom.cdf(PLAN_c, PLAN_n, ps)
    ltpd = ps[np.argmin(np.abs(pas - 0.10))]
    print(f"  LTPD (Pa = 10 %) = {ltpd:.4f}  -> {ltpd:.2%}")
    print(f"  Razon de discriminacion LTPD/AQL = {ltpd/0.01:.1f}")
    aoq = ps * pas * (LOTE_N - PLAN_n) / LOTE_N
    print(f"  AOQL = {aoq.max():.4f} ({aoq.max():.2%}) en p = {ps[np.argmax(aoq)]:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--out", type=str, default=".")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    peso = generar_peso(rng)
    atr = generar_atributos(rng)
    msa = generar_msa(rng)
    oc = curva_oc()

    peso.to_csv(out / "itata_peso.csv", index=False)
    atr.to_csv(out / "itata_atributos.csv", index=False)
    msa.to_csv(out / "itata_msa.csv", index=False)
    oc.to_csv(out / "oc_plan.csv", index=False)
    print(f"Datos didacticos generados en {out.resolve()} (semilla {args.seed})\n")
    analizar(peso, atr, msa, oc)


if __name__ == "__main__":
    main()
