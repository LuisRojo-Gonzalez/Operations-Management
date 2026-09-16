# %% [markdown]
# # Unidad 2 · Jerarquía y reconciliación
#
# **Planta:** Conservas del Itata S.A. — TOTAL → 3 familias → 6 SKU
# **Datos:** `DatosClases/2_Pronostico/itata_demanda.csv`, `itata_jerarquia.csv`
#
# ---
#
# ## Qué se decide aquí
#
# Hasta aquí se pronosticó una serie a la vez. Pero la planta no pronostica una
# serie: pronostica **un sistema de series encajadas**. Comercial mira el total,
# los jefes de línea miran las familias, y producción programa SKU. Los tres
# números tienen que ser el mismo número visto a distinta altura.
#
# Si no lo son —si el pronóstico del total no coincide con la suma de los
# pronósticos de los SKU— el problema no es estadístico sino organizacional: la
# reunión de S&OP deja de discutir qué va a pasar y pasa a discutir cuál planilla
# vale. Esa propiedad se llama **coherencia**, y no se obtiene sola: pronosticar
# cada serie por separado con el mejor método disponible produce, casi con
# certeza, un conjunto incoherente.
#
# ## Las tres respuestas clásicas
#
# | enfoque | qué hace | coherente | usa toda la información |
# |---|---|---|---|
# | **bottom-up** | pronostica los SKU y suma hacia arriba | sí | solo la del nivel inferior |
# | **top-down** | pronostica el total y lo reparte con proporciones históricas | sí | solo la del agregado |
# | **MinT** | pronostica **todos** los niveles y proyecta al espacio coherente | sí | toda |
#
# La reconciliación óptima (Wickramasuriya, Athanasopoulos y Hyndman) parte de
# los pronósticos *base* de todos los niveles —incoherentes— y los corrige con la
# proyección que minimiza la varianza del error reconciliado:
#
# $$\tilde{y} = S\,(S' W^{-1} S)^{-1} S' W^{-1}\, \hat{y}$$
#
# donde `S` es la matriz de agregación y `W` la covarianza de los errores base.
# Las variantes se distinguen solo por cómo estiman `W`: **OLS** la supone
# identidad, **WLS** usa la diagonal, y **shrink** encoge la matriz muestral
# completa hacia su diagonal. No son tres métodos distintos: son tres apuestas
# sobre cuánta estructura de correlación se puede estimar con los datos que hay.
#
# ## Cifras de referencia (WAPE a nivel de SKU)
#
# | método | WAPE |
# |---|---|
# | top-down (proporciones históricas) | **0,3621** ← el peor |
# | base (sin reconciliar, incoherente) | — |
# | bottom-up | 0,1488 |
# | **MinT-shrink** | **0,1364** ← el mejor, en todos los niveles |
#
# Que top-down sea el peor **siendo coherente** es el resultado del cuaderno, y
# la explicación está en los datos de Itata, no en la fórmula.

# %%
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comun import datos, tabla, figura, resumen, verificar, estilo

warnings.filterwarnings("ignore")
estilo()

M = 52          # periodicidad anual en semanas
H_MAX = 4       # horizontes evaluados
N_ORIGENES = 12

dem = pd.read_csv(datos("2_Pronostico", "itata_demanda.csv"))
jer = pd.read_csv(datos("2_Pronostico", "itata_jerarquia.csv"))

# El padre de TOTAL viene vacío en el CSV y pandas lo lee como NaN. Si se deja
# así, cualquier agrupación por 'padre' pierde la fila raíz o revienta con
# KeyError: nan. Se fuerza a cadena vacía antes de tocar nada más.
jer["padre"] = jer["padre"].fillna("").astype(str)
jer["serie"] = jer["serie"].astype(str)

print(jer.to_string(index=False))

# %% [markdown]
# ## 1. Construir la jerarquía y la matriz S
#
# La matriz `S` traduce «los SKU» a «todas las series». Tiene una fila por serie
# —total, familias y SKU— y una columna por SKU, y en cada fila marca con 1 los
# SKU que esa serie agrega. Las últimas filas forman la identidad: cada SKU se
# agrega a sí mismo.
#
# Con `S`, la coherencia tiene una definición algebraica y no retórica: un vector
# de pronósticos `ŷ` de todas las series es coherente si y solo si existe un
# vector `b` de pronósticos de SKU tal que `ŷ = S b`. Es decir, si `ŷ` vive en el
# espacio columna de `S`. Todo lo demás es un conjunto de números que no cierran.

# %%
SKUS = sorted(dem.sku.unique())
FAMILIAS = sorted(dem.familia.unique())
SERIES = ["TOTAL"] + FAMILIAS + SKUS

hist = {}
for s in SKUS:
    d = dem[dem.sku == s].sort_values("t")
    hist[s] = d.pedidos_cajas.to_numpy(float)
for f in FAMILIAS:
    hist[f] = sum(hist[s] for s in SKUS if s.startswith(f))
hist["TOTAL"] = sum(hist[s] for s in SKUS)

n_obs = len(hist["TOTAL"])

# matriz de agregacion
S = np.zeros((len(SERIES), len(SKUS)))
for i, serie in enumerate(SERIES):
    for j, s in enumerate(SKUS):
        if serie == "TOTAL" or serie == s or (serie in FAMILIAS and s.startswith(serie)):
            S[i, j] = 1.0

print(f"{len(SERIES)} series ({len(FAMILIAS)} familias, {len(SKUS)} SKU) · "
      f"{n_obs} semanas")
print(f"\nMatriz S  ({S.shape[0]} x {S.shape[1]}):\n")
print(pd.DataFrame(S.astype(int), index=SERIES, columns=SKUS).to_string())

# comprobacion algebraica de que S describe la agregacion real
b_real = np.array([hist[s] for s in SKUS])
y_real = np.array([hist[s] for s in SERIES])
assert np.allclose(S @ b_real, y_real), "la matriz S no reproduce las agregaciones"
print("\nS reproduce exactamente las series agregadas de los datos.")

# %% [markdown]
# ## 2. Por qué en Itata agregar NO estabiliza
#
# Antes de reconciliar nada conviene mirar los datos, porque es lo que explica
# todo lo que sigue. La creencia habitual es que agregar promedia el ruido y
# deja una serie más suave y más fácil de pronosticar. Eso es cierto cuando los
# componentes son parecidos o independientes.
#
# En Itata no lo son: las tres familias tienen su temporada alta en **meses
# distintos** (duraznos en una estación, tomates en otra, porotos en otra). Al
# sumarlas, los picos de una llenan los valles de la otra y la estacionalidad
# **se cancela**. El total queda más plano que sus partes, y esa planitud no es
# estabilidad: es información destruida.
#
# Se mide con el coeficiente de variación de la componente estacional de cada
# serie.

# %%
def perfil_estacional(y, m=M):
    """Índice estacional medio por semana del año, normalizado a media 1."""
    n_ciclos = len(y) // m
    y_rec = y[-n_ciclos * m:].reshape(n_ciclos, m)
    idx = y_rec.mean(axis=0)
    return idx / idx.mean()


filas = []
perfiles = {}
for serie in ["TOTAL"] + FAMILIAS:
    p = perfil_estacional(hist[serie])
    perfiles[serie] = p
    filas.append({"serie": serie, "amplitud_estacional_%": 100 * (p.max() - p.min()) / 2,
                  "cv_estacional": p.std() / p.mean(),
                  "semana_peak": int(np.argmax(p)) + 1})
est = pd.DataFrame(filas)
print(est.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

peaks = {r.serie: r.semana_peak for _, r in est.iterrows()}
print(f"\n  Semanas de peak: " +
      "  ".join(f"{k} = {v}" for k, v in peaks.items() if k != "TOTAL"))
print(f"  Peak del TOTAL: semana {peaks['TOTAL']}")
amp_total = float(est[est.serie == "TOTAL"]["amplitud_estacional_%"].iloc[0])
amp_fam = float(est[est.serie != "TOTAL"]["amplitud_estacional_%"].mean())
print(f"\n  Amplitud estacional del TOTAL:   {amp_total:5.1f} %")
print(f"  Amplitud media de las familias: {amp_fam:5.1f} %")
print(f"  El agregado es MAS plano que sus partes: "
      f"{amp_fam / max(amp_total, 1e-9):.2f} veces.")

# %%
fig, ax = plt.subplots(figsize=(11, 4.2))
colores = {"DUR": "#c62828", "TOM": "#1565c0", "POR": "#2e7d32"}
for f in FAMILIAS:
    ax.plot(range(1, M + 1), perfiles[f], lw=1.8, color=colores.get(f, "#666"),
            label=f"familia {f}")
ax.plot(range(1, M + 1), perfiles["TOTAL"], lw=3.0, color="#37474f",
        label="TOTAL (agregado)")
ax.axhline(1.0, color="#999", lw=0.8, ls=":")
ax.set_xlabel("semana ISO del año")
ax.set_ylabel("índice estacional (media = 1)")
ax.set_title("Las familias tienen sus picos en meses distintos: al sumarlas, "
             "la estacionalidad se cancela", loc="left")
ax.legend(ncol=4)
figura(fig, "estacionalidad_por_nivel")
plt.show()

# %% [markdown]
# El gráfico es el argumento completo. La línea gruesa —el total— es
# notoriamente **más plana** que cualquiera de las familias que la componen.
#
# Consecuencia inmediata y poco intuitiva: el agregado es una serie *pobre* en
# señal estacional. Un método que pronostique el total y reparta hacia abajo con
# proporciones fijas le está imponiendo a cada familia una estacionalidad que no
# es la suya —de hecho, una que no es la de ninguna—. Eso es exactamente lo que
# hace top-down, y por eso va a perder.

# %% [markdown]
# ## 3. Pronósticos base de todas las series
#
# La reconciliación necesita pronósticos base de **cada** serie, cada una con su
# propio método. Se usa Holt-Winters multiplicativo, que es el que salió mejor
# entre los de parámetros fijos en el cuaderno 04, aplicado por igual a los tres
# niveles para que la comparación sea limpia: lo que se compara es el método de
# reconciliación, no el método de pronóstico.
#
# Estos pronósticos base son, casi con seguridad, **incoherentes**. Es lo
# esperable y es el punto de partida, no un defecto.

# %%
def holt_winters(y_hist, h, alfa=0.2, beta=0.02, gamma=0.2, m=M):
    """Holt-Winters multiplicativo, escrito a mano (ver cuaderno 02)."""
    y = np.asarray(y_hist, float)
    if len(y) < 2 * m:
        return np.repeat(y[-1], h)
    y = np.where(y <= 0, 1e-6, y)

    n_ciclos = len(y) // m
    base = y[:n_ciclos * m].reshape(n_ciclos, m)
    nivel = base.mean()
    est = base.mean(axis=0) / nivel
    est = np.where(est <= 0, 1e-6, est)
    tend = (base[-1].mean() - base[0].mean()) / max(1, (n_ciclos - 1) * m)

    for i, v in enumerate(y):
        s = est[i % m]
        nivel_ant = nivel
        nivel = alfa * (v / s) + (1 - alfa) * (nivel + tend)
        tend = beta * (nivel - nivel_ant) + (1 - beta) * tend
        est[i % m] = gamma * (v / max(nivel, 1e-6)) + (1 - gamma) * s

    return np.array([max(0.0, (nivel + (k + 1) * tend) * est[(len(y) + k) % m])
                     for k in range(h)])


def base_todas(corte, h=H_MAX):
    """Pronóstico base de cada serie con información hasta 'corte' (exclusivo)."""
    return np.array([holt_winters(hist[s][:corte], h) for s in SERIES])


corte_demo = n_obs - H_MAX
yhat_demo = base_todas(corte_demo)
b_demo = yhat_demo[-len(SKUS):]
incoh = yhat_demo[0] - b_demo.sum(axis=0)

print("Pronóstico base en el último origen (h = 1..4):\n")
print(pd.DataFrame(yhat_demo, index=SERIES,
                   columns=[f"h={k+1}" for k in range(H_MAX)])
      .to_string(float_format=lambda v: f"{v:,.0f}"))
print(f"\n  Incoherencia del total (TOTAL − suma de SKU), por horizonte:")
for k in range(H_MAX):
    print(f"    h={k+1}: {incoh[k]:+10,.1f} cajas  "
          f"({100*incoh[k]/max(1e-9, b_demo.sum(axis=0)[k]):+.2f} %)")
print("\nNi enorme ni despreciable: suficiente para que dos áreas discutan.")

# %% [markdown]
# ## 4. Los tres reconciliadores
#
# Todos tienen la misma forma `ỹ = S G ŷ`, y se distinguen solo por `G`:
#
# * **bottom-up** ignora los niveles altos: `G = [0 | I]` toma los pronósticos de
#   SKU y descarta el resto.
# * **top-down** ignora los niveles bajos: reparte el pronóstico del total según
#   la proporción histórica de cada SKU.
# * **MinT** resuelve `G = (S' W⁻¹ S)⁻¹ S' W⁻¹` con la covarianza `W` de los
#   errores base.
#
# Una propiedad que conviene comprobar y no dar por sabida: MinT es
# **insesgado-preservante**, es decir, `S G S = S`. Si los pronósticos base ya
# fueran coherentes, MinT los dejaría intactos en vez de moverlos.

# %%
def G_bottom_up(S, n_series, n_sku):
    G = np.zeros((n_sku, n_series))
    G[:, -n_sku:] = np.eye(n_sku)
    return G


def G_top_down(S, n_series, n_sku, proporciones):
    G = np.zeros((n_sku, n_series))
    G[:, 0] = proporciones          # reparte SOLO el total
    return G


def G_mint(S, W):
    Wi = np.linalg.pinv(W)
    return np.linalg.pinv(S.T @ Wi @ S) @ S.T @ Wi


def W_ols(res):
    return np.eye(res.shape[0])


def W_wls(res):
    return np.diag(np.maximum(res.var(axis=1), 1e-8))


def W_shrink(res):
    """Covarianza muestral encogida hacia su diagonal (Schäfer-Strimmer).

    Con pocos residuos y muchas series, la covarianza muestral es singular o
    casi: invertirla amplifica ruido. El encogimiento la mezcla con su propia
    diagonal en la proporción que el propio dato sugiere.
    """
    n, T = res.shape
    Sm = np.cov(res, bias=True) if T > 1 else np.eye(n)
    D = np.diag(np.diag(Sm))
    sd = np.sqrt(np.maximum(np.diag(Sm), 1e-12))
    R = Sm / np.outer(sd, sd)
    r_off = R[~np.eye(n, dtype=bool)]
    var_r = np.var(r_off) if r_off.size else 0.0
    lam = float(np.clip(var_r / max(np.sum(r_off ** 2) / max(1, r_off.size), 1e-12),
                        0.0, 1.0))
    return lam * D + (1 - lam) * Sm


# comprobacion de la propiedad SGS = S
res_falsos = np.random.default_rng(0).normal(0, 1, (len(SERIES), 30))
for nombre, Wf in [("OLS", W_ols), ("WLS", W_wls), ("shrink", W_shrink)]:
    G = G_mint(S, Wf(res_falsos))
    err = np.abs(S @ G @ S - S).max()
    print(f"  MinT-{nombre:<7} ||S G S − S||_max = {err:.2e}")
print("\nLas tres cumplen S G S = S: si el pronóstico base ya fuera coherente,")
print("MinT no lo tocaría.")

# %% [markdown]
# ## 5. Evaluación con origen rodante
#
# Se repite el procedimiento del cuaderno 04, ahora sobre las diez series a la
# vez: en cada uno de los 12 orígenes se pronostican todos los niveles con la
# información disponible, se reconcilia con cada método, y se compara contra lo
# que realmente pasó.
#
# Los residuos que alimentan `W` se estiman **dentro de cada origen**, con la
# información de ese momento. Estimarlos con toda la serie sería exactamente la
# fuga de información del cuaderno 03, ahora disfrazada de matriz de covarianza.

# %%
def residuos_en_muestra(corte, ventana=26, h=1):
    """Errores de un paso de los pronósticos base, dentro de la muestra."""
    cols = []
    for c in range(corte - ventana, corte):
        yb = base_todas(c, h=1)[:, 0]
        real = np.array([hist[s][c] for s in SERIES])
        cols.append(real - yb)
    return np.array(cols).T


metodos = {}
props = np.array([hist[s][:n_obs - H_MAX - N_ORIGENES].sum() for s in SKUS])
props = props / props.sum()

filas = []
for origen in range(N_ORIGENES):
    corte = n_obs - H_MAX - (N_ORIGENES - 1 - origen)
    yhat = base_todas(corte, h=H_MAX)
    real = np.array([hist[s][corte:corte + H_MAX] for s in SERIES])
    res = residuos_en_muestra(corte)

    Gs = {
        "base (sin reconciliar)": None,
        "bottom-up": G_bottom_up(S, len(SERIES), len(SKUS)),
        "top-down": G_top_down(S, len(SERIES), len(SKUS), props),
        "MinT-OLS": G_mint(S, W_ols(res)),
        "MinT-WLS": G_mint(S, W_wls(res)),
        "MinT-shrink": G_mint(S, W_shrink(res)),
    }
    for nombre, G in Gs.items():
        rec = yhat if G is None else S @ (G @ yhat)
        for i, serie in enumerate(SERIES):
            nivel = ("total" if serie == "TOTAL"
                     else "familia" if serie in FAMILIAS else "sku")
            for k in range(H_MAX):
                filas.append({"metodo": nombre, "origen": origen, "serie": serie,
                              "nivel": nivel, "h": k + 1,
                              "real": real[i, k], "pred": rec[i, k]})

ev = pd.DataFrame(filas)
print(f"{len(ev):,} evaluaciones · {N_ORIGENES} orígenes × {len(SERIES)} series "
      f"× {H_MAX} horizontes × {len(Gs)} métodos")

# %% [markdown]
# ## 6. Resultados por nivel
#
# El WAPE se calcula por nivel agregando errores y realizaciones antes de
# dividir, no promediando WAPE individuales: un promedio de razones le da el
# mismo peso a un SKU chico que a uno grande, y el chico domina el resultado sin
# tener importancia operacional.

# %%
def wape_grupo(g):
    return np.abs(g.real - g.pred).sum() / max(1e-9, g.real.abs().sum())


tab = (ev.groupby(["metodo", "nivel"]).apply(wape_grupo)
         .unstack("nivel")[["total", "familia", "sku"]])
orden = ["base (sin reconciliar)", "top-down", "bottom-up",
         "MinT-OLS", "MinT-WLS", "MinT-shrink"]
tab = tab.loc[[m for m in orden if m in tab.index]]
print("WAPE por nivel\n")
print(tab.to_string(float_format=lambda v: f"{v:.4f}"))

# coherencia efectiva de cada metodo
print("\nCoherencia (|TOTAL − suma de SKU| medio, en cajas):\n")
for metodo in tab.index:
    sub = ev[ev.metodo == metodo]
    piv = sub.pivot_table(index=["origen", "h"], columns="serie", values="pred")
    err = (piv["TOTAL"] - piv[SKUS].sum(axis=1)).abs().mean()
    marca = "  <== INCOHERENTE" if err > 1e-6 else ""
    print(f"  {metodo:<24} {err:12,.4f}{marca}")

# %% [markdown]
# ### Lo que hay que leer en esta tabla
#
# Tres cosas, en este orden:
#
# 1. **El pronóstico base es incoherente** y ya se ve en la última tabla: la suma
#    de los SKU no da el total. Cualquier reconciliación lo arregla; la pregunta
#    es a qué precio en exactitud.
# 2. **Top-down es coherente y es el peor a nivel de SKU.** No falla por estar
#    mal implementado: falla porque el pronóstico del total —esa serie plana de
#    la sección 2— no contiene la estacionalidad de ninguna familia, y las
#    proporciones históricas fijas no se la devuelven. Reparte bien el nivel y
#    destruye la forma.
# 3. **MinT-shrink gana en todos los niveles a la vez**, incluido el total. Eso
#    no es obvio: uno esperaría que mejorar abajo costara arriba. No ocurre
#    porque MinT no *reparte* un número sino que usa los diez pronósticos como
#    diez mediciones ruidosas del mismo sistema, y las combina según cuánto se
#    equivoca cada una.
#
# Bottom-up queda en medio y es la respuesta defendible cuando no se confía en la
# estimación de `W`: no usa la información de los niveles altos, pero tampoco
# inventa estructura.

# %%
fig, ax = plt.subplots(figsize=(11, 4.4))
x = np.arange(len(tab.index))
ancho = 0.26
for i, (niv, color) in enumerate([("total", "#37474f"), ("familia", "#1565c0"),
                                  ("sku", "#c62828")]):
    ax.bar(x + (i - 1) * ancho, tab[niv], ancho, label=niv, color=color)
ax.set_xticks(x)
ax.set_xticklabels(tab.index, rotation=18, ha="right")
ax.set_ylabel("WAPE")
ax.set_title("Top-down es coherente y aun así el peor a nivel de SKU; "
             "MinT-shrink gana en los tres niveles", loc="left")
ax.legend(title="nivel")
figura(fig, "wape_por_nivel")
plt.show()

# %% [markdown]
# ## 7. Comprobación contra las cifras de referencia

# %%
w_sku = tab["sku"]
print("Comprobación:")
ok = [
    verificar(w_sku["top-down"], 0.3621, "WAPE SKU, top-down", tol=0.05),
    verificar(w_sku["bottom-up"], 0.1488, "WAPE SKU, bottom-up", tol=0.05),
    verificar(w_sku["MinT-shrink"], 0.1364, "WAPE SKU, MinT-shrink", tol=0.05),
]
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

# Las relaciones de orden son lo que la teoria obliga y no dependen de
# detalles de implementacion: se exigen como asercion.
assert w_sku["top-down"] > w_sku["bottom-up"], \
    "top-down deberia ser peor que bottom-up a nivel de SKU"
assert w_sku["MinT-shrink"] <= w_sku["bottom-up"] + 1e-9, \
    "MinT-shrink no deberia ser peor que bottom-up"
for metodo in ["bottom-up", "top-down", "MinT-OLS", "MinT-WLS", "MinT-shrink"]:
    sub = ev[ev.metodo == metodo]
    piv = sub.pivot_table(index=["origen", "h"], columns="serie", values="pred")
    assert (piv["TOTAL"] - piv[SKUS].sum(axis=1)).abs().max() < 1e-6, \
        f"{metodo} deberia producir pronosticos coherentes"
print("Todas las reconciliaciones son coherentes y el orden teórico se cumple.")

tabla(tab.reset_index(), "wape_por_nivel")
tabla(est, "estacionalidad_por_serie")
resumen({"wape_sku": {k: float(v) for k, v in w_sku.items()},
         "wape_total": {k: float(v) for k, v in tab["total"].items()},
         "wape_familia": {k: float(v) for k, v in tab["familia"].items()},
         "semanas_peak": {k: int(v) for k, v in peaks.items()}},
        "resumen_jerarquia")
print("\nGuardado en resultados/")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **Coherencia no es exactitud.** Top-down entrega números que cierran
#    perfectamente y son los peores de la tabla. Que la planilla cuadre no dice
#    nada sobre si el pronóstico sirve.
# 2. **«Agregar siempre estabiliza» es falso** cuando los componentes tienen
#    fases opuestas. En Itata el total es la serie con *menos* señal estacional,
#    y cualquier método que dependa de pronosticarlo bien arranca perdiendo.
# 3. **MinT usa todos los niveles como mediciones del mismo sistema.** Por eso
#    puede mejorar arriba y abajo a la vez, algo que ni bottom-up ni top-down
#    pueden ofrecer.
# 4. **`W` se estima con la información del origen, no con toda la serie.** La
#    fuga de información del cuaderno 03 reaparece aquí disfrazada de matriz de
#    covarianza, y es más difícil de ver.
#
# ## Ejercicios
#
# 1. Reemplace las proporciones históricas de top-down por las de las últimas 26
#    semanas. ¿Mejora? ¿Alcanza a bottom-up? Explique qué parte del problema
#    corrige y cuál no.
# 2. Corra MinT-shrink estimando `W` con **toda** la serie en vez de con la
#    ventana previa a cada origen. El WAPE debería mejorar. Explique por qué esa
#    mejora es ficticia y qué se está midiendo en realidad.
# 3. Construya una jerarquía alternativa agrupando por **formato** (480 g contra
#    820 g y 3000 g) en vez de por familia. ¿Cambia el ranking de métodos? ¿Qué
#    dice eso sobre elegir la jerarquía antes de elegir el método?
