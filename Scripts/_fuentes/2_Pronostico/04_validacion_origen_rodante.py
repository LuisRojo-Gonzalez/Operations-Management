# %% [markdown]
# # Unidad 2 · Validación con origen rodante
#
# **Planta:** Conservas del Itata S.A. — DUR-480 (suave) y POR-3000 (grumosa)
# **Datos:** `DatosClases/2_Pronostico/itata_demanda.csv`
#
# ---
#
# ## Qué se decide aquí
#
# Los cuadernos 02 y 03 compararon métodos. Este decide **si esas comparaciones
# significan algo**, y entrega el número que la Unidad 3 necesita para dimensionar
# el inventario.
#
# Hay una diferencia que conviene tener clara desde el principio. Medir el error
# de un pronóstico no es medir el ajuste de un modelo a sus datos. El ajuste
# siempre se puede mejorar agregando parámetros; el error de pronóstico no. Y la
# única forma de estimar el error de pronóstico es **reproducir el procedimiento
# completo tal como se va a ejecutar en la planta**: todas las semanas, con la
# información disponible ese lunes, re-estimando el modelo.
#
# | decisión | opción cómoda | opción correcta | qué cuesta equivocarse |
# |---|---|---|---|
# | cómo evaluar | una partición 80/20 | 12 orígenes consecutivos | el ranking depende de qué semana quedó como corte |
# | cuándo re-estimar | una vez, con toda la serie | en cada origen, con `y[:origen]` | el modelo ya vio el futuro al elegir sus parámetros |
# | qué reportar | el error | el error **y** el sesgo | un pronóstico insesgado puede ser pésimo; uno sesgado rompe el stock de seguridad |
# | qué σ usar en la Unidad 3 | el residuo del ajuste | σ del error de **validación**, por horizonte | el stock de seguridad queda corto y el servicio no se cumple |
#
# La última fila es el acoplamiento entre unidades. El entregable de esta unidad
# no es «un pronóstico»: es un pronóstico **y** una σ̂(i, h), y la segunda entra
# directo al modelo de planificación de la Unidad 3.
#
# ## Cifras de referencia (DUR-480, 12 orígenes, h = 1..4)
#
# | método | WAPE |
# |---|---|
# | **ETS** (parámetros estimados por verosimilitud) | **0,1904** |
# | Holt-Winters multiplicativo (parámetros fijos) | 0,1951 |
# | media móvil(4) | 0,2151 |
# | Holt amortiguado (φ = 0,9) | 0,2170 |
# | SES(α = 0,3) | 0,2293 |
# | **ingenuo estacional (la base)** | **0,2858** |
# | ingenuo | 0,3288 |

# %%
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comun import datos, tabla, figura, resumen, verificar, estilo

# statsmodels avisa de convergencia al reajustar ETS en series cortas
warnings.filterwarnings("ignore")
estilo()

dem = pd.read_csv(datos("2_Pronostico", "itata_demanda.csv"))

_m = (dem.sku == "TOM-480") & (dem.t == 87)          # limpieza del cuaderno 01
for c in ("pedidos_cajas", "despachos_cajas"):
    dem.loc[_m, c] = dem.loc[_m, c] / 24

M = 52
H_MAX, N_ORIGENES = 4, 12


def serie(sku, col="pedidos_cajas"):
    g = dem[dem.sku == sku].sort_values("t")
    return g[col].to_numpy(dtype=float)


y_suave = serie("DUR-480")
y_grumosa = serie("POR-3000")
print(f"DUR-480 : n = {len(y_suave)}   media = {y_suave.mean():,.0f} cajas")
print(f"POR-3000: n = {len(y_grumosa)}   media = {y_grumosa.mean():,.0f} cajas   "
      f"ceros = {(y_grumosa == 0).mean():.1%}")

# %% [markdown]
# ## 1. Por qué una sola partición no basta
#
# El procedimiento habitual es cortar la serie en un punto, entrenar antes y
# evaluar después. Da un número, y un número parece un resultado. El problema es
# que ese número tiene una varianza que nadie reporta: **depende de qué semana
# quedó como corte**, y la serie tiene estacionalidad, así que distintos cortes
# caen en fases distintas del ciclo.
#
# La demostración es directa: se corre exactamente la misma comparación entre dos
# métodos moviendo el corte una semana a la vez, y se mira qué tanto se mueve el
# resultado.

# %%
def wape(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    s = np.abs(y_real).sum()
    return float(np.abs(y_real - y_pred).sum() / s) if s else float("nan")


def holt_winters(y_hist, h, alfa=0.2, beta=0.02, gamma=0.2, m=M):
    """Holt-Winters multiplicativo del cuaderno 02, en versión compacta."""
    y_hist = np.asarray(y_hist, dtype=float)
    n_ciclos = len(y_hist) // m
    if n_ciclos < 2:
        raise ValueError("Holt-Winters necesita dos ciclos completos")
    prom = np.array([y_hist[i * m:(i + 1) * m].mean() for i in range(n_ciclos)])
    l = prom[0]
    b = (prom[-1] - prom[0]) / max(1, (n_ciclos - 1) * m)
    s = np.array([np.mean([y_hist[c * m + i] / prom[c] for c in range(n_ciclos)
                           if prom[c] > 0]) for i in range(m)])
    s = list(s / s.mean())
    for t, v in enumerate(y_hist):
        st = s[t % m] if t < m else s[-m]
        l_ant = l
        l = alfa * (v / (st if abs(st) > 1e-9 else 1.0)) + (1 - alfa) * (l + b)
        s.append(gamma * (v / l if l else 1.0) + (1 - gamma) * st)
        b = beta * (l - l_ant) + (1 - beta) * b
    return np.array([(l + i * b) * s[-m + (i - 1) % m] for i in range(1, h + 1)])


def media_movil(y_hist, h, k=4):
    return np.repeat(np.asarray(y_hist, float)[-k:].mean(), h)


print("La MISMA comparación, cambiando solo dónde se corta la serie:\n")
print(f"{'corte en t':>12}{'Holt-Winters':>16}{'media móvil(4)':>18}   gana")
gana_hw = 0
filas_corte = []
for corte in range(167, 179):
    real = y_suave[corte:corte + H_MAX]
    a = wape(real, holt_winters(y_suave[:corte], H_MAX))
    b = wape(real, media_movil(y_suave[:corte], H_MAX))
    gana_hw += int(a < b)
    filas_corte.append({"corte": corte, "Holt-Winters": a, "media móvil(4)": b})
    print(f"{corte:>12}{a:>16.4f}{b:>18.4f}   "
          f"{'Holt-Winters' if a < b else 'media móvil'}")

cortes_df = pd.DataFrame(filas_corte).set_index("corte")
print(f"\nHolt-Winters gana en {gana_hw} de los 12 cortes.")
print(f"WAPE de Holt-Winters según el corte: mínimo {cortes_df['Holt-Winters'].min():.4f}, "
      f"máximo {cortes_df['Holt-Winters'].max():.4f} "
      f"(factor {cortes_df['Holt-Winters'].max()/cortes_df['Holt-Winters'].min():.1f}x)")

# %% [markdown]
# El rango es enorme: el mismo método, sobre la misma serie, con el mismo código,
# mide entre 0,09 y 0,33 según en qué semana se haya cortado. Y el ganador
# **cambia** de corte en corte.
#
# Hay que ser preciso sobre qué muestra esto y qué no. No muestra que un método
# sea mejor que el otro: muestra que **con una sola partición no se puede
# saber**. Quien reporta «mi modelo tiene 12 % de WAPE» sin decir cuántos cortes
# midió está reportando el mejor tiro de doce, y probablemente sin saberlo.
#
# La salida no es buscar un corte «representativo» —no existe— sino promediar
# sobre varios. Doce orígenes dan doce mediciones a cada horizonte, y con eso se
# puede decir si la diferencia entre dos métodos sobrevive al cambio de semana.

# %% [markdown]
# ## 2. El origen rodante, y la fuga que evita
#
# El procedimiento es el que haría la planta si guardara registro. Para cada
# origen *o* en una secuencia de semanas consecutivas:
#
# 1. se toma **solo** `y[:o]` —lo que se sabía ese lunes—;
# 2. se **re-estima el modelo desde cero** con esos datos;
# 3. se pronostica h = 1, 2, 3, 4 semanas;
# 4. se guardan los cuatro pares (real, pronosticado).
#
# Doce orígenes por cuatro horizontes dan 48 pares por método, y el WAPE se
# calcula sobre el conjunto. Los orígenes se eligen consecutivos y al final de la
# serie a propósito: son las semanas más parecidas a la que viene.
#
# El punto 2 es el que se salta casi todo el mundo. Ajustar el modelo una vez con
# las 182 semanas y después «validar» sobre las últimas 48 observaciones es una
# fuga de información completa: los parámetros del modelo —el α de SES, los
# índices estacionales de Holt-Winters, los coeficientes de una regresión— se
# eligieron **viendo** los datos sobre los que después se lo evalúa. El error que
# sale de ahí no es un error de pronóstico; es un residuo de ajuste con otro
# nombre.
#
# Para que la re-estimación sea obligatoria y no un acto de disciplina, la función
# recibe una **fábrica** —un `callable` que devuelve un modelo nuevo— y no una
# instancia. Así es imposible arrastrar el estado de un origen al siguiente, y la
# fuga deja de depender de que el analista se acuerde.

# %%
def origen_rodante(y, fabrica, h_max=H_MAX, n_origenes=N_ORIGENES, paso=1):
    """Una fila por (origen, h) con el valor real y el pronosticado.

    `fabrica` es un callable SIN argumentos que devuelve un pronosticador
    nuevo: una función que recibe (historia, h) y devuelve h valores. Se exige
    la fábrica para garantizar que cada origen reajusta desde cero.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    minimo = n - (n_origenes - 1) * paso - h_max
    filas = []
    for k in range(n_origenes):
        corte = minimo + k * paso
        h = min(h_max, n - corte)
        if h <= 0:
            continue
        try:
            pred = fabrica()(y[:corte], h)
        except Exception as e:               # un método que no puede ajustarse
            filas.append({"origen": corte, "h": np.nan, "y": np.nan,
                          "yhat": np.nan, "error": str(e)[:60]})
            continue
        for i in range(h):
            filas.append({"origen": corte, "h": i + 1, "y": float(y[corte + i]),
                          "yhat": float(pred[i]), "error": ""})
    return pd.DataFrame(filas)


MINIMO = len(y_suave) - (N_ORIGENES - 1) - H_MAX
print(f"Orígenes: t = {MINIMO} a {MINIMO + N_ORIGENES - 1}")
print(f"El primer origen entrena con {MINIMO} semanas "
      f"({MINIMO / M:.1f} ciclos anuales) y el último con "
      f"{MINIMO + N_ORIGENES - 1}.")
print(f"Total de pares (real, pronóstico) por método: "
      f"{N_ORIGENES * H_MAX}")

# %% [markdown]
# ## 3. Los métodos, incluido uno nuevo: ETS
#
# Los métodos del cuaderno 02 vuelven aquí envueltos en funciones
# `(historia, h) -> pronóstico`. Se agrega uno nuevo, y la diferencia es
# conceptual, no de código.
#
# Hasta ahora los parámetros de suavizamiento se fijaron a mano: α = 0,2,
# β = 0,02, γ = 0,2. Son valores razonables y son **inventados**. **ETS**
# (*Error, Trend, Seasonal*) es la misma familia de modelos escrita como un modelo
# estadístico con espacio de estados, lo que permite estimar α, β y γ **por máxima
# verosimilitud** en vez de elegirlos. Aquí se usa la implementación de
# `statsmodels`, que es lo correcto: la recursión ya se entendió en el cuaderno
# 02; lo que agrega ETS es el optimizador, y reescribir un optimizador a mano no
# enseña nada sobre pronóstico.
#
# Dos salvaguardas que ETS necesita y que muestran por qué «automático» es una
# palabra optimista:
#
# * la estacionalidad **multiplicativa no está definida con ceros**. En POR-3000,
#   con 35 % de semanas en cero, el ajuste diverge y devuelve pronósticos de seis
#   cifras. Hay que caer a aditiva, y hay que dejar constancia de que se cayó.
# * con **menos de dos ciclos completos** no hay con qué estimar 52 índices
#   estacionales. Hay que ajustar sin componente estacional.
#
# Ninguna de las dos las hace la librería sola. Un `fit()` sin estas guardas
# entrega números que parecen pronósticos.

# %%
def ingenuo(y_hist, h):
    return np.repeat(np.asarray(y_hist, float)[-1], h)


def ingenuo_estacional(y_hist, h, m=M):
    ult = np.asarray(y_hist, float)[-m:]
    return np.array([ult[i % m] for i in range(h)])


def ses(y_hist, h, alfa=0.3):
    l = float(y_hist[0])
    for v in np.asarray(y_hist, float)[1:]:
        l = alfa * v + (1 - alfa) * l
    return np.repeat(l, h)


def holt(y_hist, h, alfa=0.3, beta=0.05, phi=0.9):
    y_hist = np.asarray(y_hist, float)
    l, b = y_hist[0], (y_hist[1] - y_hist[0]) if len(y_hist) > 1 else 0.0
    for v in y_hist[1:]:
        l_ant = l
        l = alfa * v + (1 - alfa) * (l + phi * b)
        b = beta * (l - l_ant) + (1 - beta) * phi * b
    return np.array([l + sum(phi ** k for k in range(1, i + 1)) * b
                     for i in range(1, h + 1)])


def croston(y_hist, h, alfa=0.1, sba=True):
    z = q = None
    intervalo = 1
    for v in np.asarray(y_hist, float):
        if v > 0:
            if z is None:
                z, q = float(v), float(intervalo)
            else:
                z = alfa * v + (1 - alfa) * z
                q = alfa * intervalo + (1 - alfa) * q
            intervalo = 1
        else:
            intervalo += 1
    tasa = 0.0 if z is None else z / q * ((1 - alfa / 2.0) if sba else 1.0)
    return np.repeat(tasa, h)


AVISOS = []


def ets(y_hist, h, m=M, tendencia="add", estacional="mul", amortiguada=True):
    """ETS con parámetros estimados por máxima verosimilitud."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_hist = np.asarray(y_hist, dtype=float)
    est = estacional
    if est is not None and len(y_hist) < 2 * m:
        est, aviso = None, f"sin estacionalidad: {len(y_hist)} < 2 x {m}"
    elif est == "mul" and (y_hist <= 0).any():
        est, aviso = "add", "estacionalidad aditiva: la serie tiene ceros"
    else:
        aviso = ""
    try:
        res = ExponentialSmoothing(
            y_hist, trend=tendencia, damped_trend=amortiguada, seasonal=est,
            seasonal_periods=m if est else None,
            initialization_method="estimated").fit(optimized=True)
    except Exception:
        res = ExponentialSmoothing(y_hist, trend=None, seasonal=None,
                                   initialization_method="estimated").fit()
        aviso = "ajuste degradado a suavizamiento simple"
    pred = np.asarray(res.forecast(h), dtype=float)
    # Un pronóstico varios órdenes de magnitud fuera del rango histórico no es
    # un pronóstico: es una divergencia numérica. Se recorta y se deja rastro.
    tope = 5.0 * max(y_hist.max(), 1.0)
    if not np.isfinite(pred).all() or (np.abs(pred) > tope).any():
        aviso = "pronóstico recortado (divergencia numérica)"
        pred = np.clip(np.nan_to_num(pred, nan=float(np.median(y_hist))), 0.0, tope)
    if aviso:
        AVISOS.append(aviso)
    return pred


METODOS_SUAVE = {
    "ingenuo": lambda: ingenuo,
    "ingenuo estacional": lambda: ingenuo_estacional,
    "media móvil(4)": lambda: media_movil,
    "SES(a=0.3)": lambda: ses,
    "Holt(phi=0.9)": lambda: holt,
    "Holt-Winters mult.": lambda: holt_winters,
    "ETS": lambda: ets,
}

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def parametros_ets(y_hist, m=M):
    r = ExponentialSmoothing(y_hist, trend="add", damped_trend=True, seasonal="mul",
                             seasonal_periods=m,
                             initialization_method="estimated").fit(optimized=True)
    return r, {k: float(r.params[k]) for k in
               ("smoothing_level", "smoothing_trend", "smoothing_seasonal",
                "damping_trend")}


print("Lo que ETS estima por máxima verosimilitud (en el cuaderno 02 estos")
print("valores se fijaron a mano en alfa = 0,20, beta = 0,02, gamma = 0,20):\n")
print(f"{'sku':<12}{'alfa':>9}{'beta':>9}{'gamma':>9}{'phi':>9}")
for s in ("DUR-480", "POR-480", "TOM-480"):
    _, p = parametros_ets(serie(s))
    print(f"{s:<12}{p['smoothing_level']:>9.4f}{p['smoothing_trend']:>9.4f}"
          f"{p['smoothing_seasonal']:>9.4f}{p['damping_trend']:>9.4f}")

_r, _p = parametros_ets(y_suave)

# %% [markdown]
# Los tres ceros de DUR-480 y POR-480 no son un error de convergencia: son el
# óptimo, y dicen algo sobre la serie que vale más que cualquier WAPE.
#
# α = 0 significa que el nivel **no se actualiza** con las observaciones nuevas:
# se queda en el valor inicial estimado y evoluciona solo por la tendencia
# amortiguada. γ = 0 significa lo mismo para los índices estacionales. Traducido:
# la verosimilitud dice que en estas dos series **la variación semanal es ruido
# puro**, y que reaccionar a ella —que es lo que hace un α de 0,20— es empeorar
# el pronóstico. El modelo estimado es, en el fondo, una curva determinista:
# nivel con deriva amortiguada, por un patrón estacional fijo.
#
# Y es exactamente lo que hay en los datos. El generador construyó estas series
# como nivel × estacionalidad × ruido lognormal independiente, sin componente
# aleatoria acumulativa. Un procedimiento estadístico bien hecho **recuperó la
# estructura del proceso que generó los datos**, sin que nadie se la dijera. Esa
# es también la razón por la que ETS le gana a Holt-Winters por tan poco: el
# margen de mejora sobre unos parámetros razonables era pequeño.
#
# TOM-480 es el contraste que da sentido a lo anterior: ahí sí sale α = 0,067,
# distinto de cero. TOM-480 es la serie del **quiebre estructural** del cuaderno
# 01 —el contrato de supermercado en t = 120—, y una serie con un escalón de
# nivel obliga al modelo a poder moverse. La verosimilitud distinguió sola las
# series que hay que seguir de las que hay que promediar.
#
# La advertencia que hay que dejar anotada: **α = 0 es una solución de borde**, y
# significa que el modelo, tal como quedó estimado, no reaccionaría a un quiebre
# futuro. Es la respuesta correcta al pasado y una apuesta sobre el futuro. Si
# mañana DUR-480 gana un contrato, ese modelo va a tardar en enterarse, y
# conviene saberlo antes y no después.

# %% [markdown]
# ## 4. Las métricas, y por qué el sesgo va aparte
#
# El cuaderno 02 ya mostró por qué MAPE no sirve con ceros —no está definida, es
# asimétrica y no pondera por volumen— y que el ranking que produce es distinto
# del de WAPE. Aquí se recuerda con una línea de código sobre POR-3000 y se pasa
# a lo que falta: **el sesgo**.
#
# WAPE, MAE y RMSE miden **magnitud** del error, con el valor absoluto o el
# cuadrado adentro. Por construcción no distinguen un error de +200 cajas de uno
# de −200 cajas. Para el inventario esa distinción lo es todo:
#
# * un error **sin signo dominante** se absorbe con stock de seguridad. Es para lo
#   que existe el stock de seguridad, y su tamaño lo fija la σ del error.
# * un error **con signo persistente** no se absorbe: se acumula. Si el pronóstico
#   queda corto un 8 % todas las semanas, en un tiempo de reposición de cuatro
#   semanas faltan 4 × 8 % = 32 % de una semana de demanda, y eso **se come el
#   stock de seguridad** que se había dimensionado para otra cosa. El nivel de
#   servicio calculado no se cumple, y el modelo de la Unidad 3 no tiene cómo
#   saberlo porque se le entregó una σ que no incluye el sesgo.
#
# De ahí la regla: **el error y el sesgo se reportan siempre juntos**, nunca uno
# solo. Y la recíproca también importa, porque es menos obvia: **un pronóstico
# insesgado puede ser pésimo**. Sesgo cero significa que los errores se cancelan
# en promedio, no que sean chicos. La celda siguiente construye los dos casos.

# %%
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m]))) if m.any() else float("nan")


def mase(y, yhat, y_entrenamiento, m=M):
    ytr = np.asarray(y_entrenamiento, float)
    if len(ytr) <= m:
        m = 1
    escala = np.mean(np.abs(ytr[m:] - ytr[:-m]))
    return float(mae(y, yhat) / escala) if escala else float("nan")


def sesgo(y, yhat):
    """Error medio. Positivo = se SUBESTIMÓ la demanda."""
    return float(np.mean(np.asarray(y, float) - np.asarray(yhat, float)))


def sesgo_relativo(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    s = np.abs(y).sum()
    return float((y - yhat).sum() / s) if s else 0.0


# los dos casos que confunden error con sesgo
ev_ets = origen_rodante(y_suave, METODOS_SUAVE["ETS"])
real, base_pred = ev_ets.y.to_numpy(), ev_ets.yhat.to_numpy()
signos = np.where(np.arange(len(real)) % 2 == 0, 1.0, -1.0)
insesgado_malo = base_pred + 0.45 * signos * base_pred     # errores que se cancelan
# escalado para que el sesgo relativo sea exactamente +8 % (queda corto)
c = (real.sum() - 0.08 * np.abs(real).sum()) / base_pred.sum()
sesgado = base_pred * c

print(f"{'pronóstico':<34}{'WAPE':>9}{'sesgo rel.':>13}{'sesgo (cajas/sem)':>20}")
for nombre, p in [("ETS tal cual", base_pred),
                  ("insesgado pero pésimo (±45 %)", insesgado_malo),
                  ("sesgado 8 % a la baja", sesgado)]:
    print(f"{nombre:<34}{wape(real, p):>9.4f}{sesgo_relativo(real, p):>13.4f}"
          f"{sesgo(real, p):>20,.1f}")

LEAD_TIME = 4
demanda_sem = float(real.mean())
faltante = 0.08 * demanda_sem * LEAD_TIME
print(f"\nEl segundo tiene un sesgo del mismo orden que ETS y un WAPE "
      f"{wape(real, insesgado_malo)/wape(real, base_pred):.1f} veces peor:")
print("  insesgado NO quiere decir bueno; quiere decir que los errores se")
print("  cancelan en promedio, no que sean chicos.")
print(f"\nEl tercero es el peligroso, y lo es justamente porque en la tabla de")
print(f"errores se ve BIEN: su WAPE ({wape(real, sesgado):.4f}) es incluso algo mejor que el de")
print(f"ETS ({wape(real, base_pred):.4f}). Un comité que elija por WAPE lo prefiere. Pero queda")
print(f"corto un 8 % todas las semanas, y en un tiempo de reposición de "
      f"{LEAD_TIME} semanas")
print(f"eso acumula {faltante:,.0f} cajas de faltante — "
      f"{faltante / demanda_sem:.2f} semanas de demanda.")
print("La sección 6 lo compara contra el stock de seguridad, que se había")
print("dimensionado para cubrir otra cosa.")

print(f"\nMAPE sobre POR-3000, que tiene {(y_grumosa == 0).mean():.0%} de semanas en cero:")
ev_g = origen_rodante(y_grumosa, lambda: croston)
n_cero = int((ev_g.y == 0).sum())
print(f"  de los {len(ev_g)} pares evaluados, {n_cero} tienen y = 0 y MAPE los")
print(f"  DESCARTA: se calcula sobre {len(ev_g) - n_cero}, y justo sobre los")
print(f"  fáciles. MAPE = {mape(ev_g.y, ev_g.yhat):.4f}, "
      f"WAPE = {wape(ev_g.y, ev_g.yhat):.4f}.")

# %% [markdown]
# ## 5. La comparación completa
#
# Ahora sí: los siete métodos sobre los doce orígenes, con las cuatro métricas
# que hay que reportar juntas.

# %%
filas, crudo = [], {}
# La escala de MASE se calcula sobre la muestra de ENTRENAMIENTO, y se usa la
# misma para todos los métodos: la historia disponible en el último origen.
y_entrena = y_suave[:len(y_suave) - H_MAX]
for nombre, fab in METODOS_SUAVE.items():
    ev = origen_rodante(y_suave, fab)
    crudo[nombre] = ev
    filas.append({"método": nombre, "WAPE": wape(ev.y, ev.yhat),
                  "MASE": mase(ev.y, ev.yhat, y_entrena), "MAE": mae(ev.y, ev.yhat),
                  "RMSE": rmse(ev.y, ev.yhat),
                  "sesgo rel.": sesgo_relativo(ev.y, ev.yhat)})

res = pd.DataFrame(filas).set_index("método").sort_values("WAPE")
print("DUR-480 — 12 orígenes, h = 1..4, 48 pares por método")
print(res.round(4).to_string())

base_wape = res.loc["ingenuo estacional", "WAPE"]
mejor = res.index[0]
print(f"\nBase (ingenuo estacional): WAPE = {base_wape:.4f}")
print(f"Mejor método: {mejor} -> mejora {1 - res.WAPE.iloc[0]/base_wape:.1%}")
peores = [k for k in res.index if res.loc[k, "WAPE"] > base_wape]
print(f"NO superan la base: {', '.join(peores) if peores else 'ninguno'}")
print(f"\nAvisos emitidos por ETS durante la validación: "
      f"{len(AVISOS) if AVISOS else 'ninguno'}")

# %% [markdown]
# ETS le gana a Holt-Winters por 0,005 de WAPE, que es **medio punto porcentual**.
# Vale la pena decir en voz alta lo que eso significa: estimar α, β y γ por máxima
# verosimilitud, en vez de fijarlos con criterio, mejoró el error un 2,4 %. En una
# serie con esta estacionalidad, los parámetros «razonables» del cuaderno 02 ya
# estaban cerca del óptimo.
#
# La diferencia que sí importa está más abajo en la tabla: **entre modelar la
# estacionalidad y no modelarla**. SES, Holt y la media móvil —los tres sin
# componente estacional— quedan entre 0,215 y 0,229; los dos que la modelan, en
# 0,19. Y la base, el ingenuo estacional, en 0,286.
#
# La lectura de MASE ahorra la discusión sobre si un WAPE de 0,19 es bueno: el
# ingenuo estacional tiene MASE ≈ 0,97, o sea que en validación se comporta
# aproximadamente como su propia definición, y ETS tiene 0,65. Un 35 % mejor que
# repetir el año pasado. Esa es la frase que se lleva a la reunión.
#
# ### Sobre la reproducibilidad de estas cifras
#
# Las cifras de este cuaderno reproducen las de referencia hasta el cuarto
# decimal, pero conviene saber cuáles son frágiles y cuáles no. Los métodos
# escritos a mano —ingenuo, ingenuo estacional, media móvil, SES, Holt,
# Holt-Winters— son aritmética determinista: van a dar el mismo número en
# cualquier máquina y cualquier versión. **ETS no**: sus parámetros salen de un
# optimizador numérico, y dependen de la inicialización que elija `statsmodels`,
# del algoritmo de optimización y de la versión de la librería. Por eso la
# comprobación de ETS de la sección 8 se hace con una tolerancia de 5 · 10⁻³ y no
# con la de los otros. Si su corrida da 0,192 en vez de 0,1904, no hay nada roto;
# si da 0,25, sí.

# %% [markdown]
# ## 6. Sigma por horizonte: lo que se le entrega a la Unidad 3
#
# Aquí está el acoplamiento entre unidades, y es lo más importante del cuaderno.
#
# El modelo de planificación de la Unidad 3 necesita un stock de seguridad. La
# fórmula estándar para un tiempo de reposición de *L* períodos es
#
# $$ss = z_{\alpha}\,\hat\sigma_L$$
#
# donde σ_L es la desviación del error de pronóstico **de la demanda acumulada en
# el tiempo de reposición**. Tres decisiones se esconden en esa letra griega, y
# las tres se toman aquí, no en la Unidad 3:
#
# **Primera: σ del error de pronóstico, no de la demanda.** Son cosas distintas.
# La desviación de la demanda mide cuánto varía la demanda; la del error mide
# cuánto de esa variación el modelo **no supo anticipar**. Un modelo que capture
# bien la estacionalidad tiene un error mucho menos variable que la demanda, y
# usar la σ de la demanda sobredimensiona el inventario en la misma proporción.
#
# **Segunda: medida en validación, no en el ajuste.** El residuo dentro de muestra
# siempre es menor que el error de pronóstico, porque el modelo ajustó sus
# parámetros para minimizarlo. Usarlo da un stock de seguridad optimista y un
# nivel de servicio que no se cumple: el número está mal en la dirección peligrosa.
#
# **Tercera: por horizonte.** El error a cuatro semanas no es el error a una
# semana. Lo que hace falta para el stock de seguridad es la σ del error
# **acumulado** sobre las L semanas del tiempo de reposición, y esa se estima
# directamente: para cada origen se suman los cuatro reales, se suman los cuatro
# pronósticos y se mira la desviación de la diferencia.

# %%
def por_horizonte(ev):
    ev = ev.dropna(subset=["y", "yhat"]).copy()
    ev["e"] = ev.y - ev.yhat
    filas = []
    for h, gg in ev.groupby("h"):
        filas.append({"h": int(h), "n": len(gg), "WAPE": wape(gg.y, gg.yhat),
                      "sesgo": gg.e.mean(), "sigma_error": gg.e.std(ddof=1)})
    return pd.DataFrame(filas)


ph = por_horizonte(crudo[mejor])
print(f"Error de {mejor} por horizonte (DUR-480):")
print(ph.round(3).to_string(index=False))

# sigma del error ACUMULADO sobre las 4 semanas de reposición
acum = crudo[mejor].groupby("origen").agg(y=("y", "sum"), yhat=("yhat", "sum"))
e_acum = (acum.y - acum.yhat).to_numpy()
sigma_L = float(np.std(e_acum, ddof=1))
sigma_1 = float(ph.loc[ph.h == 1, "sigma_error"].iloc[0])
sigma_raiz = sigma_1 * np.sqrt(H_MAX)

# sigma del residuo DENTRO DE MUESTRA, para contrastar
_aj = ExponentialSmoothing(y_suave, trend="add", damped_trend=True, seasonal="mul",
                           seasonal_periods=M,
                           initialization_method="estimated").fit(optimized=True)
sigma_ajuste = float(np.std(y_suave - _aj.fittedvalues, ddof=1))

z = 1.645                                     # nivel de servicio del 95 %
print(f"\nStock de seguridad para L = {H_MAX} semanas y z = {z} (95 % de servicio):\n")
print(f"{'σ usada':<44}{'σ (cajas)':>12}{'ss = z·σ':>12}")
for etiqueta, s in [
        ("residuo del AJUSTE dentro de muestra (mal)", sigma_ajuste),
        ("σ(h=1) escalada por raíz de L (aproximación)", sigma_raiz),
        ("σ del error ACUMULADO en validación (bien)", sigma_L)]:
    print(f"{etiqueta:<44}{s:>12,.0f}{z * s:>12,.0f}")

print(f"\nUsar el residuo del ajuste deja el stock de seguridad en "
      f"{z * sigma_ajuste:,.0f} cajas")
print(f"en vez de {z * sigma_L:,.0f}: un {1 - sigma_ajuste/sigma_L:.0%} menos. "
      "Ese descuento no se")
print("nota en ningún indicador del modelo de pronóstico; se nota en el quiebre.")

# ¿por qué el error acumulado NO es sigma(1) por raiz de L?
ancho = crudo[mejor].pivot_table(index="origen", columns="h", values="y") - \
    crudo[mejor].pivot_table(index="origen", columns="h", values="yhat")
corr12 = float(np.corrcoef(ancho[1], ancho[2])[0, 1])
print(f"\nLa aproximación σ(h=1)·√L da {z * sigma_raiz:,.0f}, un "
      f"{sigma_raiz/sigma_L - 1:.0%} más que el valor medido.")
print(f"Esa fórmula supone errores independientes entre semanas, y aquí la")
print(f"correlación entre el error a h = 1 y el de h = 2, dentro del mismo")
print(f"origen, es {corr12:+.2f}: los errores se CANCELAN parcialmente. Es lo que")
print("pasa cuando el modelo desfasa el peak estacional una semana en vez de")
print("equivocarse en el nivel: sobra una semana y falta la siguiente, y en el")
print("acumulado del tiempo de reposición eso importa mucho menos.")

print(f"\nY el sesgo de la sección 4, puesto al lado: un pronóstico 8 % corto")
print(f"acumula {faltante:,.0f} cajas de faltante en las mismas {LEAD_TIME} semanas,")
print(f"que es el {faltante / (z * sigma_L):.0%} del stock de seguridad completo.")
print("Un sesgo que en la tabla de errores no se veía se lleva casi todo el")
print("colchón que se había calculado para cubrir la variabilidad.")

# %% [markdown]
# ### La tabla σ̂(i, h) que entra a la Unidad 3
#
# El entregable no es una σ sino una matriz: un valor por SKU y horizonte. Se
# calcula con el mismo procedimiento para los seis SKU, usando en cada uno el
# método que le corresponde según el cuadrante ADI/CV² del cuaderno 01 —ETS para
# las series suaves, SBA para la grumosa—, que es la razón por la que esa
# clasificación no era un adorno.

# %%
filas = []
for sku in sorted(dem.sku.unique()):
    ys = serie(sku)
    grumosa = (ys == 0).mean() > 0.2
    fab = (lambda: croston) if grumosa else (lambda: ets)
    ev = origen_rodante(ys, fab)
    p = por_horizonte(ev)
    fila = {"sku": sku, "método": "SBA" if grumosa else "ETS",
            "media (cajas/sem)": ys.mean(),
            "WAPE": wape(ev.y, ev.yhat)}
    for _, r in p.iterrows():
        fila[f"sigma h={int(r.h)}"] = r.sigma_error
    ac = ev.groupby("origen").agg(y=("y", "sum"), yhat=("yhat", "sum"))
    fila["sigma acumulada L=4"] = float((ac.y - ac.yhat).std(ddof=1))
    filas.append(fila)

sigmas = pd.DataFrame(filas).set_index("sku")
print("σ̂(i, h) — el insumo de la Unidad 3:")
print(sigmas.to_string(float_format=lambda v: f"{v:,.1f}",
                       formatters={"WAPE": "{:.4f}".format}))
print("\nLéase la columna WAPE junto a la de la media: POR-3000 tiene el peor")
print("WAPE del catálogo y la σ más chica en cajas. No es una contradicción:")
print("es un SKU de bajo volumen y demanda grumosa. Se gestiona por INVENTARIO")
print("—con esa sigma— y no por pronóstico, que es lo que el cuaderno 02")
print("anticipó al ver su cuadrante.")

# %% [markdown]
# ## 7. Los tres gráficos que cierran la unidad

# %%
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5))

# --- (a) la varianza de una sola partición
a = ax[0, 0]
a.plot(cortes_df.index, cortes_df["Holt-Winters"], "o-", color="#c62828", lw=1.8,
       ms=5, label="Holt-Winters")
a.plot(cortes_df.index, cortes_df["media móvil(4)"], "s-", color="#1565c0", lw=1.8,
       ms=5, label="media móvil(4)")
a.axhline(res.loc["Holt-Winters mult.", "WAPE"], color="#c62828", ls="--", lw=1.2)
a.axhline(res.loc["media móvil(4)", "WAPE"], color="#1565c0", ls="--", lw=1.2)
a.text(cortes_df.index[-1], res.loc["Holt-Winters mult.", "WAPE"] - 0.022,
       "WAPE de los 12 orígenes juntos", fontsize=8, color="#37474f", ha="right")
a.set_xlabel("semana en que se corta (una sola partición)")
a.set_ylabel("WAPE de las 4 semanas siguientes")
a.set_title("(a) Con una partición, el ganador depende del corte", loc="left")
a.legend(fontsize=8)

# --- (b) el ranking de los siete métodos
b = ax[0, 1]
orden = res.sort_values("WAPE", ascending=False)
colores = ["#2e7d32" if k in ("ETS", "Holt-Winters mult.")
           else ("#9e9e9e" if k == "ingenuo estacional" else "#78909c")
           for k in orden.index]
b.barh(range(len(orden)), orden.WAPE.to_numpy(), color=colores)
b.axvline(base_wape, color="#c62828", lw=1.4, ls="--")
b.text(base_wape + 0.006, len(orden) - 1.1, "base:\ningenuo estacional",
       fontsize=8, color="#c62828", va="top")
for i, (k, v) in enumerate(orden.WAPE.items()):
    b.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8)
b.set_yticks(range(len(orden)))
b.set_yticklabels(orden.index, fontsize=8.5)
b.set_xlim(0, 0.40)
b.set_xlabel("WAPE (12 orígenes, h = 1..4)")
b.set_title("(b) Verde: los dos que modelan la estacionalidad", loc="left")

# --- (c) los 12 orígenes dibujados
c_ax = ax[1, 0]
tt = np.arange(1, len(y_suave) + 1)
c_ax.plot(tt[-40:], y_suave[-40:], "o-", color="#37474f", lw=1.6, ms=4,
          label="demanda real", zorder=3)
ev_m = crudo[mejor]
for j, (o, gg) in enumerate(ev_m.groupby("origen")):
    c_ax.plot(gg.origen + gg.h, gg.yhat, "o-", color="#c62828", lw=1.5, ms=2.5,
              alpha=0.9, label=f"{mejor}, cada origen" if j == 0 else None)
    c_ax.plot([o + 0.5], [y_suave[-40:].min()], "^", color="#1565c0", ms=6,
              label="origen" if j == 0 else None)
c_ax.set_xlabel("semana t")
c_ax.set_ylabel("cajas")
c_ax.set_title("(c) Doce pronósticos de 4 semanas, uno por origen", loc="left")
c_ax.legend(fontsize=8, loc="upper left")

# --- (d) sigma por horizonte y el error acumulado
d_ax = ax[1, 1]
d_ax.bar(ph.h - 0.0, ph.sigma_error, width=0.55, color="#1565c0",
         label="σ del error a horizonte h")
d_ax.axhline(sigma_ajuste, color="#c62828", lw=1.6, ls="--",
             label=f"residuo del ajuste ({sigma_ajuste:,.0f})")
d_ax.axhline(sigma_L, color="#2e7d32", lw=1.6,
             label=f"σ del error acumulado L = 4 ({sigma_L:,.0f})")
for _, r in ph.iterrows():
    d_ax.text(r.h, r.sigma_error + 8, f"{r.sigma_error:,.0f}", ha="center",
              fontsize=8)
d_ax.set_xticks(ph.h.to_numpy())
d_ax.set_xlabel("horizonte h (semanas)")
d_ax.set_ylabel("σ del error (cajas)")
d_ax.set_ylim(0, float(ph.sigma_error.max()) * 1.45)
d_ax.set_title("(d) La línea roja es la que subdimensiona el inventario",
               loc="left")
d_ax.legend(fontsize=8, loc="upper left")

figura(fig, "origen_rodante")
plt.show()

# %% [markdown]
# El panel (c) es el que explica el método sin palabras: doce trazos rojos cortos,
# cada uno arrancando en su propio origen y cubriendo cuatro semanas. Ninguno usa
# información posterior a su punto de partida, y el WAPE de la tabla es el error
# de los cuarenta y ocho puntos rojos tomados juntos.
#
# El panel (d) tiene una irregularidad que **no** conviene esconder: la σ no crece
# monótonamente con el horizonte. Sube en h = 4, pero h = 3 tiene menos dispersión
# que h = 1. Teóricamente debería crecer, porque a mayor horizonte hay más
# incertidumbre acumulada. Lo que pasa es que cada σ̂(h) se estima con **doce
# observaciones**, o sea once grados de libertad, y a ese tamaño de muestra el
# ruido de estimación es del mismo orden que la tendencia que se quiere ver. La
# conclusión práctica no es «la σ no crece»; es que **con doce orígenes no se
# puede afirmar la forma de la curva**, y si la Unidad 3 necesita σ(h) para
# horizontes largos hay que correr más orígenes o suavizar la curva estimada.

# %% [markdown]
# ## 8. Comprobación contra las cifras de referencia

# %%
print("Comprobación:")
ok = [
    verificar(res.loc["ETS", "WAPE"], 0.1904, "WAPE de ETS (el mejor)", tol=5e-3),
    verificar(res.loc["Holt-Winters mult.", "WAPE"], 0.1951,
              "WAPE de Holt-Winters multiplicativo"),
    verificar(res.loc["media móvil(4)", "WAPE"], 0.2151, "WAPE de la media móvil(4)"),
    verificar(res.loc["SES(a=0.3)", "WAPE"], 0.2293, "WAPE de SES(0.3)"),
    verificar(res.loc["ingenuo estacional", "WAPE"], 0.2858,
              "WAPE del ingenuo estacional (la base)"),
    verificar(res.loc["ingenuo", "WAPE"], 0.3288, "WAPE del ingenuo"),
    verificar(res.loc["ingenuo estacional", "MASE"], 0.9744,
              "MASE del ingenuo estacional"),
    verificar(res.loc["ETS", "MASE"], 0.6491, "MASE de ETS", tol=5e-3),
    verificar(len(ev_ets), N_ORIGENES * H_MAX, "pares (real, pronóstico) por método"),
]
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

assert res.index[0] == "ETS", f"el mejor método debía ser ETS, salió {res.index[0]}"
assert res.index[-1] == "ingenuo", "el peor método debía ser el ingenuo"
assert sigma_ajuste < sigma_L, \
    "el residuo del ajuste debe ser MENOR que el error de validación acumulado"
assert wape(real, insesgado_malo) > 2 * wape(real, base_pred), \
    "el pronóstico insesgado construido debía ser mucho peor que ETS"
print("\nLa tolerancia de ETS es 5e-3 y no 1e-3 porque sus parámetros salen de un")
print("optimizador numérico: dependen de la versión de statsmodels y de la")
print("inicialización. Los seis métodos escritos a mano son deterministas.")

tabla(res.reset_index(), "origen_rodante_dur480")
tabla(ph, "sigma_por_horizonte")
tabla(sigmas.reset_index(), "sigma_por_sku_horizonte")
resumen({"wape": {k: float(v) for k, v in res.WAPE.items()},
         "mejor": mejor, "base": float(base_wape),
         "sigma_h": {int(r.h): float(r.sigma_error) for _, r in ph.iterrows()},
         "sigma_acumulada_L4": sigma_L, "sigma_residuo_ajuste": sigma_ajuste,
         "stock_seguridad_z1645": z * sigma_L}, "resumen_origen_rodante")
print("\nGuardado en resultados/ — sigma_por_sku_horizonte.csv es el insumo de la")
print("Unidad 3.")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **Una sola partición no mide nada.** El mismo método, sobre la misma serie,
#    da WAPE entre 0,09 y 0,33 según en qué semana se corte, y el ganador cambia
#    de corte en corte. Un error de pronóstico sin la cantidad de orígenes al lado
#    es un número sin interpretación.
# 2. **En cada origen se re-estima desde cero.** Ajustar una vez con toda la serie
#    y validar sobre el final es la fuga más común de la unidad: los parámetros se
#    eligieron viendo los datos de prueba. Pedir una fábrica de modelos en vez de
#    un modelo convierte esa disciplina en algo que el código garantiza.
# 3. **El sesgo se reporta aparte del error, siempre.** Un pronóstico insesgado
#    puede ser pésimo —los errores se cancelan, no desaparecen— y uno con poco
#    error pero sesgado a la baja se come el stock de seguridad de manera
#    acumulativa: 8 % semanal por cuatro semanas de reposición es un tercio de
#    semana de demanda que nadie presupuestó.
# 4. **El entregable a la Unidad 3 es σ̂(i, h), medida en validación.** El residuo
#    del ajuste dentro de muestra es sistemáticamente menor y produce un stock de
#    seguridad optimista. El error no aparece en ningún indicador del modelo de
#    pronóstico: aparece en el quiebre de stock tres meses después.
#
# ## Ejercicios
#
# 1. Corra la validación con `n_origenes` igual a 4, 12 y 30 sobre DUR-480.
#    ¿Cambia el ranking de los siete métodos? ¿Cuánto se mueve el WAPE de ETS?
#    Fíjese también en cuánta historia recibe el primer origen en cada caso y
#    explique el compromiso que hay entre número de orígenes y longitud mínima de
#    entrenamiento.
# 2. Modifique `origen_rodante` para que el modelo se ajuste **una sola vez** con
#    toda la serie y después pronostique sobre los 12 orígenes. Compare el WAPE de
#    ETS bajo ese procedimiento con el de este cuaderno. ¿Cuánto «mejora» el
#    modelo por hacer trampa, y por qué la mejora es mayor en ETS que en el
#    ingenuo estacional?
# 3. Con la tabla `sigma_por_sku_horizonte.csv`, calcule el stock de seguridad de
#    los seis SKU para z = 1,645 y L = 4 semanas, y conviértalo a pesos usando
#    `precio_clp_caja`. ¿Qué fracción del capital inmovilizado corresponde a
#    POR-3000, y qué fracción de las cajas? Discuta si ese SKU debería seguir
#    gestionándose con el mismo nivel de servicio que los demás.
