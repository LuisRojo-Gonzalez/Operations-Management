# %% [markdown]
# # Unidad 2 · El pronóstico como problema de regresión supervisada
#
# **Planta:** Conservas del Itata S.A. — DUR-480 (la serie suave del cuaderno 02)
# **Datos:** `DatosClases/2_Pronostico/itata_demanda.csv`
#
# ---
#
# ## Qué se decide aquí
#
# Un modelo de regresión no sabe que los datos son una serie de tiempo. Le da lo
# mismo el orden de las filas. Eso tiene una consecuencia buena y una fatal.
#
# La buena: si se logra escribir el problema como una tabla `X → y`, se abre de
# golpe todo el catálogo de `scikit-learn`, y con él la posibilidad de meter
# covariables —promociones, feriados, precio— que los métodos de suavizamiento
# del cuaderno 02 simplemente **no pueden ver**.
#
# La fatal: si el orden de las filas no importa, nada impide que la validación
# cruzada entrene con el futuro y evalúe sobre el pasado. El error que sale de
# ahí es una ficción, y es una ficción **optimista**, que es la peor clase.
#
# Así que aquí se deciden tres cosas:
#
# | decisión | opción cómoda | opción correcta | qué cuesta equivocarse |
# |---|---|---|---|
# | cómo armar la tabla `X → y` | rezagos de 1 período para todo horizonte | un modelo por horizonte, con rezagos desplazados en *h* | se usa información que en *t* no existía |
# | cómo validar | `KFold(shuffle=True)`, que es el default mental | `TimeSeriesSplit` | el error estimado no tiene relación con el de producción |
# | a qué atribuir la mejora | «el aprendizaje automático es mejor» | separar el aporte del **algoritmo** del de la **información** | se compra un modelo caro para conseguir lo que daba una regresión lineal |
#
# La tercera es la que deja la lección de este cuaderno. La comparación honesta
# no es «Holt-Winters contra RandomForest»: es **a igualdad de información**
# primero, y **a igualdad de algoritmo** después.
#
# ## Cifras de referencia (DUR-480, 12 orígenes, h = 1..4)
#
# | cantidad | valor |
# |---|---|
# | ingenuo estacional (la base) | WAPE 0,2858 |
# | Holt-Winters multiplicativo (mejor clásico del cuaderno 02) | WAPE 0,1951 |
# | mejor modelo de `sklearn` **sin** covariables de calendario | 0,2317 — **peor que Holt-Winters** |
# | mejor modelo de `sklearn` **con** covariables de calendario | 0,1490 |
# | Ridge con covariables (una regresión lineal) | 0,1671 — mejor que RandomForest |
# | efecto de `min_samples_leaf` por defecto en HistGB | 0,1490 → 0,2335 |
# | MAE que reporta `KFold` aleatorio contra `TimeSeriesSplit` | 217 contra 392 (Ridge) |

# %%
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comun import datos, tabla, figura, resumen, verificar, estilo

# sklearn y statsmodels avisan de convergencia en series cortas; los avisos se
# silencian para que la salida sea legible, no porque no importen.
warnings.filterwarnings("ignore")
estilo()

dem = pd.read_csv(datos("2_Pronostico", "itata_demanda.csv"))

# limpieza mínima del cuaderno 01, igual que en el 02
_m = (dem.sku == "TOM-480") & (dem.t == 87)
for c in ("pedidos_cajas", "despachos_cajas"):
    dem.loc[_m, c] = dem.loc[_m, c] / 24

M = 52
SKU = "DUR-480"
g = dem[dem.sku == SKU].sort_values("t").reset_index(drop=True)
y = g.pedidos_cajas.to_numpy(dtype=float)
ys = pd.Series(y, index=g.index)

print(f"{SKU}: n = {len(y)} semanas   media = {y.mean():,.0f} cajas")
print(f"semanas con promoción     : {int(g.promo.sum())} de {len(g)} "
      f"({g.promo.mean():.1%})")
print(f"semanas de Fiestas Patrias: {int(g.fiestas_patrias.sum())}")
print(f"semanas de Semana Santa   : {int(g.semana_santa.sum())}")

# %% [markdown]
# ## 1. De serie de tiempo a tabla de regresión
#
# La conversión consiste en inventar columnas. Una serie `y_1, ..., y_n` no tiene
# variables explicativas; hay que construirlas, y cada familia de columnas
# codifica una hipótesis distinta sobre qué hace que la demanda de una semana sea
# la que es.
#
# **Rezagos.** `y_{t-1}`, `y_{t-2}`, ... codifican la inercia: la demanda de esta
# semana se parece a la de la semana pasada. Es lo mismo que hace SES, pero en
# vez de imponer pesos que decaen geométricamente, se dejan libres y el modelo los
# estima. Más flexible, y también más parámetros que estimar con la misma data.
#
# **Medias móviles.** Resumen varias semanas en una columna. Aportan poco que los
# rezagos no tengan, salvo estabilidad: una media de 13 semanas tiene mucho menos
# ruido que cualquier rezago individual.
#
# **Términos de Fourier.** Aquí está la decisión importante. Una estacionalidad
# semanal de período m = 52 se puede codificar con 51 variables indicadoras —una
# por semana ISO— o con K pares de senos y cosenos:
#
# $$\sin\!\left(\frac{2\pi k w}{52}\right),\ \cos\!\left(\frac{2\pi k w}{52}\right),
# \quad k = 1, \dots, K$$
#
# Con K = 3 son **6 columnas en vez de 51**. Con 182 semanas de historia, 51
# indicadoras significan estimar un parámetro cada 3,5 observaciones: eso no es
# estimar, es memorizar. Los términos de Fourier imponen que el ciclo anual sea
# suave, que es una suposición razonable para demanda de consumo y que reduce el
# problema a seis números.
#
# **Indicadores de calendario.** `promo`, `fiestas_patrias`, `semana_santa`,
# `planta_cerrada`. Son las columnas que ningún método del cuaderno 02 puede
# usar, y —adelantando el final— son las que van a decidir la comparación. Nótese
# que las cuatro son **conocidas de antemano**: la promoción la planifica
# comercial con semanas de anticipación y los feriados están en el calendario. Si
# no lo fueran, habría que pronosticarlas primero, y el error de ese pronóstico se
# sumaría al del modelo.
#
# **Tendencia.** El índice `t` como variable continua. Sirve para Ridge y es
# peligrosa en los árboles: un árbol no extrapola, así que para cualquier `t`
# mayor que el máximo del entrenamiento devuelve la hoja del extremo. No rompe
# nada aquí porque el horizonte es corto, pero es la razón por la que un bosque
# aleatorio nunca va a capturar una tendencia sostenida.
#
# ### La parte que se hace mal: el desplazamiento por horizonte
#
# El error más común al armar esta tabla es usar `y_{t-1}` para pronosticar a
# cuatro semanas. En el origen *t* se conoce `y_t`, no `y_{t+3}`; un modelo que
# prediga `y_{t+4}` usando `y_{t+3}` está usando un dato que todavía no existe.
#
# La solución que se usa aquí es la **estrategia directa**: un modelo distinto
# para cada horizonte, y en el modelo de horizonte *h* el rezago más reciente es
# `y_{t-h}`. Cuatro horizontes, cuatro modelos. La alternativa —la estrategia
# recursiva, que pronostica a un paso y realimenta su propio pronóstico— es más
# económica pero acumula el error de cada paso en el siguiente.

# %%
BINARIAS = ["promo", "fiestas_patrias", "semana_santa", "planta_cerrada"]


def construir_features(g, h, K=3, m=M, con_calendario=True):
    """Tabla X para pronosticar a horizonte h. Una fila por semana objetivo.

    Todas las columnas de la fila de la semana objetivo T tienen que ser
    conocidas en el origen T - h. Por eso los rezagos parten en h y no en 1.
    """
    X = pd.DataFrame(index=g.index)
    s = g.pedidos_cajas.astype(float)

    for j in range(4):                       # y[T-h], y[T-h-1], y[T-h-2], y[T-h-3]
        X[f"rezago_{h + j}"] = s.shift(h + j)
    X["rezago_52"] = s.shift(m)              # el mismo período del año anterior

    X["media_movil_4"] = s.shift(h).rolling(4).mean()
    X["media_movil_13"] = s.shift(h).rolling(13).mean()
    X["media_movil_52"] = s.shift(h).rolling(m).mean()

    w = g.iso_semana.to_numpy(dtype=float)   # posición en el ciclo anual
    for k in range(1, K + 1):
        X[f"seno_{k}"] = np.sin(2 * np.pi * k * w / m)
        X[f"coseno_{k}"] = np.cos(2 * np.pi * k * w / m)

    X["tendencia"] = g.t.astype(float)

    if con_calendario:
        for c in BINARIAS:
            X[c] = g[c].astype(float)
    return X


X1 = construir_features(g, h=1)
validas = X1.notna().all(axis=1)
print(f"Columnas construidas: {X1.shape[1]}")
print(f"Filas utilizables: {int(validas.sum())} de {len(X1)} "
      f"— las primeras {int((~validas).sum())} se pierden por el rezago de 52")
print(f"\nPrimeras columnas de la tabla (h = 1), tres filas del medio:")
print(X1.loc[100:102, ["rezago_1", "rezago_52", "media_movil_13", "seno_1",
                       "coseno_1", "promo", "fiestas_patrias"]]
      .round(2).to_string())

print("\nVerificación del desplazamiento — la fila objetivo t = 101 (índice 100):")
print(f"  rezago_1 de la fila 100  = {X1.loc[100, 'rezago_1']:,.0f}")
print(f"  y de la semana anterior  = {y[99]:,.0f}   (deben coincidir)")
X4 = construir_features(g, h=4)
print(f"  con h = 4, el rezago más reciente es 'rezago_4' = "
      f"{X4.loc[100, 'rezago_4']:,.0f} = y[96] = {y[96]:,.0f}")
print("  y NO existe ninguna columna con y[97], y[98] ni y[99]: en el origen")
print("  t = 97 esos valores todavía no se habían observado.")

# %% [markdown]
# ## 2. Por qué la validación cruzada aleatoria es inválida
#
# `KFold(shuffle=True)` reparte las filas al azar en cinco bloques, entrena con
# cuatro y evalúa en el quinto. Sobre datos independientes es lo correcto. Sobre
# una serie de tiempo hace dos cosas ilegítimas a la vez:
#
# 1. **Entrena con el futuro y predice el pasado.** Cuatro quintos de las semanas
#    posteriores a la semana evaluada están en el conjunto de entrenamiento. El
#    modelo ya vio hacia dónde iba la serie.
# 2. **Rompe la dependencia temporal a favor del modelo.** Las filas vecinas
#    comparten casi todos sus rezagos. Si la semana 100 está en entrenamiento y la
#    101 en prueba, el modelo ya vio `y_100` como *variable objetivo* y ahora la
#    recibe otra vez como `rezago_1`. No está pronosticando: está recordando.
#
# El resultado es un error de validación artificialmente bajo. Y el problema no
# es que sea bajo: es que **no tiene ninguna relación monótona** con el error de
# producción, así que tampoco sirve para elegir entre modelos. Un procedimiento de
# selección que se apoya en él elige mal y además no avisa.
#
# `TimeSeriesSplit` respeta el orden: el bloque *k* de prueba es siempre posterior
# a todo lo que hay en entrenamiento. Cada partición es una simulación en pequeño
# de lo que va a pasar en producción.
#
# La demostración que sigue corre **el mismo modelo, sobre los mismos datos**, y
# cambia únicamente el esquema de partición.

# %%
ENTRENA_HASTA = 166          # índice de la última semana observada en el 1er origen


def preprocesador(columnas):
    """Estandariza lo continuo y deja pasar las indicadoras.

    Va DENTRO del Pipeline a propósito: si se estandariza la tabla completa
    antes de partir, la media y la desviación usadas en entrenamiento se
    calcularon con los datos de prueba. Es una fuga pequeña y silenciosa, y
    `Pipeline` la elimina por construcción porque reajusta el escalador en cada
    partición.
    """
    binarias = [c for c in BINARIAS if c in columnas]
    continuas = [c for c in columnas if c not in BINARIAS]
    return ColumnTransformer([("continuas", StandardScaler(), continuas),
                              ("binarias", "passthrough", binarias)])


def modelos_sklearn(columnas):
    """Los tres candidatos, cada uno dentro de su Pipeline."""
    return {
        "Ridge": Pipeline([
            ("pre", preprocesador(columnas)),
            ("mdl", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([
            ("pre", preprocesador(columnas)),
            ("mdl", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                          random_state=0))]),
        "HistGB": Pipeline([
            ("pre", preprocesador(columnas)),
            ("mdl", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                                  max_depth=3, min_samples_leaf=5,
                                                  random_state=0))]),
    }


X_cv = X1[validas & (X1.index <= ENTRENA_HASTA)]
y_cv = ys[X_cv.index]
print(f"Validación cruzada sobre {len(X_cv)} semanas de ENTRENAMIENTO "
      f"(t = {int(g.t[X_cv.index[0]])} a {int(g.t[X_cv.index[-1]])}).")
print("Las 15 semanas finales no entran aquí: son la prueba de la sección 4.\n")

aleatorio = KFold(5, shuffle=True, random_state=0)
temporal = TimeSeriesSplit(5)
temporal_largo = TimeSeriesSplit(5, test_size=6)

print("Tamaño del conjunto de entrenamiento en cada partición:")
print(f"  KFold aleatorio          : "
      f"{[len(tr) for tr, _ in aleatorio.split(X_cv)]}")
print(f"  TimeSeriesSplit(5)       : "
      f"{[len(tr) for tr, _ in temporal.split(X_cv)]}")
print(f"  TimeSeriesSplit(test=6)  : "
      f"{[len(tr) for tr, _ in temporal_largo.split(X_cv)]}\n")

filas_cv = []
for nombre, mdl in modelos_sklearn(X_cv.columns).items():
    def mae_cv(cv_obj):
        return -cross_val_score(mdl, X_cv, y_cv, cv=cv_obj,
                                scoring="neg_mean_absolute_error").mean()
    a, t, tl = mae_cv(aleatorio), mae_cv(temporal), mae_cv(temporal_largo)
    filas_cv.append({"modelo": nombre, "MAE KFold aleatorio": a,
                     "MAE TimeSeriesSplit": t,
                     "MAE TSS (test=6)": tl,
                     "subestimación": 1 - a / t})

cv = pd.DataFrame(filas_cv).set_index("modelo")
print(cv.round(3).to_string())
print("\n'subestimación' = cuánto menos error reporta el KFold aleatorio que la")
print("partición temporal, sobre exactamente los mismos datos y el mismo modelo.")
print(f"\nRanking según KFold aleatorio  : "
      f"{list(cv['MAE KFold aleatorio'].sort_values().index)}")
print(f"Ranking según TimeSeriesSplit  : "
      f"{list(cv['MAE TimeSeriesSplit'].sort_values().index)}")

# %% [markdown]
# El KFold aleatorio reporta entre un 35 % y un 45 % menos de error que la
# partición temporal, con los mismos datos y el mismo modelo. Ese es el efecto
# que había que demostrar y está demostrado: la cifra que sale de una validación
# cruzada aleatoria sobre una serie de tiempo es artificialmente buena.
#
# Y la brecha **no es la misma para todos los modelos** —va de 1,55 a 1,80 veces—,
# lo que es peor que si fuera constante: un sesgo constante se descuenta, uno que
# depende del modelo **cambia el ranking**. De hecho lo cambia aquí: el KFold
# aleatorio deja a `Ridge` primero y la partición temporal deja a `RandomForest`.
# Un procedimiento de selección construido sobre el primero elige otro modelo.
#
# ### La otra mitad de la historia, que casi nunca se cuenta
#
# `TimeSeriesSplit` es correcto en su lógica, pero mirar los tamaños de
# entrenamiento que imprimió la celda anterior obliga a matizar el número: con 5
# particiones sobre 115 filas, la **primera entrena con 20 semanas**. Veinte
# semanas no alcanzan ni para media temporada; ese pliegue no mide el error del
# modelo, mide el error de un modelo sin datos, y arrastra el promedio hacia
# arriba. Por eso la tercera columna, con particiones de prueba de tamaño fijo y
# entrenamientos de 85 a 109 semanas, da un error sistemáticamente menor.
#
# La lección completa es entonces de dos partes, y la segunda es la que distingue
# a quien entendió el problema:
#
# 1. el KFold aleatorio está **mal por construcción**, porque usa el futuro;
# 2. `TimeSeriesSplit` está bien por construcción, pero **su nivel depende de cómo
#    se diseñen los pliegues**, y hay que mirar cuánta historia recibe cada uno.
#
# Ninguna de las dos cifras es todavía «el error que se va a observar en la
# planta». Para eso hace falta el protocolo de la sección siguiente —orígenes
# consecutivos con una ventana mínima de entrenamiento decente—, que es lo que el
# cuaderno 04 formaliza con el nombre de validación de origen rodante.
#
# Para ver la estructura de las particiones conviene mirarlas dibujadas; el panel
# (a) de la figura de la sección 6 lo hace.

# %% [markdown]
# ## 3. Los métodos clásicos, sobre los mismos orígenes
#
# Para comparar hace falta que los dos lados resuelvan el mismo problema. El
# protocolo es el que formaliza el cuaderno 04: **12 orígenes** consecutivos y en
# cada uno un pronóstico a **h = 1, 2, 3 y 4** semanas, re-estimando el modelo en
# cada origen. Son 48 pares (real, pronosticado) por método, y el WAPE se calcula
# sobre el conjunto.
#
# Los dos referentes son el ingenuo estacional —la base que todo modelo tiene que
# aprobar— y Holt-Winters multiplicativo, que fue el mejor método clásico del
# cuaderno 02. Se reimplementan aquí en versión compacta para que el cuaderno sea
# autocontenido; son las mismas recursiones del cuaderno 02.

# %%
def holt_winters(y_hist, h, alfa=0.2, beta=0.02, gamma=0.2, m=M):
    """Holt-Winters multiplicativo. Ajusta sobre y_hist y devuelve h pasos."""
    y_hist = np.asarray(y_hist, dtype=float)
    if len(y_hist) < 2 * m:
        raise ValueError("Holt-Winters necesita dos ciclos completos")
    n_ciclos = len(y_hist) // m
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


def ingenuo_estacional(y_hist, h, m=M):
    ult = np.asarray(y_hist, dtype=float)[-m:]
    return np.array([ult[i % m] for i in range(h)])


H_MAX, N_ORIGENES = 4, 12
CORTES = [len(y) - (N_ORIGENES - 1) - H_MAX + k for k in range(N_ORIGENES)]
print(f"Orígenes: t = {CORTES[0]} a {CORTES[-1]}  ({N_ORIGENES} orígenes)")
print(f"En cada uno se pronostican h = 1..{H_MAX} semanas -> "
      f"{N_ORIGENES * H_MAX} pares (real, pronóstico) por método.")


def wape(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    s = np.abs(y_real).sum()
    return float(np.abs(y_real - y_pred).sum() / s) if s else float("nan")


filas = []
for corte in CORTES:
    hist = y[:corte]
    pred = {"Holt-Winters mult.": holt_winters(hist, H_MAX),
            "ingenuo estacional": ingenuo_estacional(hist, H_MAX)}
    for nombre, p in pred.items():
        for i in range(H_MAX):
            filas.append({"modelo": nombre, "familia": "clásico", "origen": corte,
                          "h": i + 1, "y": y[corte + i], "yhat": float(p[i])})
ev_clasicos = pd.DataFrame(filas)

for nombre, gg in ev_clasicos.groupby("modelo"):
    print(f"  {nombre:<22} WAPE = {wape(gg.y, gg.yhat):.4f}")

# %% [markdown]
# ## 4. La comparación, hecha dos veces
#
# Aquí está el diseño del experimento, y es lo único que hay que entender de esta
# sección. Los tres modelos de `sklearn` se evalúan **dos veces sobre los mismos
# 12 orígenes**:
#
# * **sin covariables de calendario**: solo rezagos, medias móviles, Fourier y
#   tendencia. Es decir, exactamente la misma información que tiene Holt-Winters:
#   el pasado de la serie y su posición en el ciclo anual.
# * **con covariables de calendario**: lo anterior más `promo`,
#   `fiestas_patrias`, `semana_santa` y `planta_cerrada`.
#
# La diferencia entre las dos corridas mide el aporte de la **información**. La
# diferencia entre la primera corrida y Holt-Winters mide el aporte del
# **algoritmo**. Correr una sola vez y comparar contra el clásico mezcla las dos
# cosas, y es la comparación que se publica en la mayoría de los informes.
#
# En cada origen y cada horizonte el modelo se reajusta desde cero con los datos
# disponibles hasta ese origen. Son 3 modelos × 2 escenarios × 4 horizontes × 12
# orígenes = 288 ajustes, y por eso el cuaderno se restringe a **un solo SKU**.

# %%
filas = []
for con_cal in (True, False):
    etiqueta = "con calendario" if con_cal else "sin calendario"
    for h in range(1, H_MAX + 1):
        X = construir_features(g, h, con_calendario=con_cal)
        utiles = X.notna().all(axis=1)
        for corte in CORTES:
            i_obj = corte + h - 1                  # índice 0-based de la semana meta
            entrena = utiles & (X.index <= corte - 1)
            for nombre, mdl in modelos_sklearn(X.columns).items():
                mdl.fit(X[entrena], ys[entrena])
                yhat = float(mdl.predict(X.iloc[[i_obj]])[0])
                filas.append({"modelo": f"{nombre} ({etiqueta})",
                              "familia": etiqueta, "origen": corte, "h": h,
                              "y": y[i_obj], "yhat": yhat})

ev_ml = pd.DataFrame(filas)
ev = pd.concat([ev_clasicos, ev_ml], ignore_index=True)

res = []
for nombre, gg in ev.groupby("modelo", sort=False):
    res.append({"modelo": nombre, "familia": gg.familia.iloc[0],
                "WAPE": wape(gg.y, gg.yhat),
                "MAE": float(np.abs(gg.y - gg.yhat).mean()),
                "sesgo rel.": float((gg.y - gg.yhat).sum() / gg.y.abs().sum())})
res = pd.DataFrame(res).set_index("modelo").sort_values("WAPE")
print(res.round(4).to_string())

base = wape(*ev_clasicos[ev_clasicos.modelo == "ingenuo estacional"][["y", "yhat"]]
            .to_numpy().T)
hw = wape(*ev_clasicos[ev_clasicos.modelo == "Holt-Winters mult."][["y", "yhat"]]
          .to_numpy().T)
sin_cal = res[res.familia == "sin calendario"].WAPE
con_cal = res[res.familia == "con calendario"].WAPE
print(f"\nBase (ingenuo estacional)        : {base:.4f}")
print(f"Holt-Winters multiplicativo      : {hw:.4f}")
print(f"Mejor sklearn SIN calendario     : {sin_cal.min():.4f}  "
      f"({sin_cal.idxmin()})")
print(f"Mejor sklearn CON calendario     : {con_cal.min():.4f}  "
      f"({con_cal.idxmin()})")

# %% [markdown]
# ### Lo que dicen los números, sin adornos
#
# **A igualdad de información, los tres modelos de `sklearn` pierden.** El mejor
# de ellos sin covariables queda en 0,232 contra 0,195 de Holt-Winters: un 19 %
# peor. Apenas le ganan al ingenuo estacional. No es un accidente de esta corrida,
# y las razones son estructurales:
#
# * **Hay 182 semanas, apenas 3,5 ciclos anuales.** Después de descartar las 52
#   primeras filas —que no tienen rezago de un año— quedan unas 115 filas de
#   entrenamiento para 15 o 19 columnas. Holt-Winters describe la misma serie con
#   tres parámetros (α, β, γ) y un vector estacional que se actualiza
#   recursivamente. Con esta cantidad de datos, la estructura impuesta vale más
#   que la flexibilidad.
# * **La estacionalidad es fuerte y regular.** El generador puso un ciclo anual
#   multiplicativo de amplitud fija. Eso es exactamente lo que la tercera ecuación
#   de Holt-Winters está diseñada para capturar, y es lo que un árbol tiene que
#   reconstruir a fuerza de cortes sobre `seno_1` y `coseno_1`.
# * **Es una sola serie.** El aprendizaje automático en pronóstico gana cuando hay
#   cientos o miles de series que comparten patrones: un modelo global aprende la
#   forma del ciclo una vez y la aplica a todas, y entonces las 115 filas se
#   convierten en 115 × N. Con un SKU no hay nada que compartir.
#
# **Con covariables, los tres ganan.** Y aquí viene lo que hay que no
# malinterpretar: **la mejora no la produjo el aprendizaje automático, la produjo
# la información**. La prueba está en la propia tabla: `Ridge`, que es una
# regresión lineal con penalización —el modelo menos «automático» del conjunto—
# le gana a `RandomForest`. Si la ventaja viniera de capturar no linealidades, el
# orden sería el inverso.
#
# Lo que hay detrás es que el generador inyecta promociones con un *lift* de
# 1,45× a 1,90× en una de cada diez semanas, y Fiestas Patrias con 1,28×. Esos
# saltos son enormes y **perfectamente conocidos de antemano**. Cualquier modelo
# que pueda leer la columna `promo` los va a acertar; ninguno que no pueda leerla
# los va a acertar jamás, por sofisticado que sea.
#
# La conclusión operativa para la planta no es «hay que usar aprendizaje
# automático». Es: **conseguir el calendario promocional de comercial vale más
# que cambiar de algoritmo**. Y si se consigue, un modelo clásico también puede
# usarlo —una regresión con errores ARIMA y regresores exógenos hace justamente
# eso—, así que ni siquiera obliga a cambiar de familia de métodos.

# %%
print("El aporte separado, en puntos de WAPE:\n")
print(f"{'modelo':<16}{'sin calendario':>16}{'con calendario':>16}{'aporte info.':>15}")
for nombre in ["Ridge", "RandomForest", "HistGB"]:
    a = res.loc[f"{nombre} (sin calendario)", "WAPE"]
    b = res.loc[f"{nombre} (con calendario)", "WAPE"]
    print(f"{nombre:<16}{a:>16.4f}{b:>16.4f}{a - b:>15.4f}")
print(f"{'Holt-Winters':<16}{hw:>16.4f}{'—':>16}{'—':>15}")
print("\nDos lecturas:")
print("  * columna 'sin calendario': los tres están por encima de 0,195, o sea")
print("    los tres pierden contra Holt-Winters con la misma información.")
print("  * columna 'aporte info.': la mejora por agregar cuatro columnas del")
print("    calendario es de 8 a 9 puntos de WAPE, varias veces la diferencia")
print("    entre cualquier par de algoritmos.")

# %% [markdown]
# ## 5. Un hiperparámetro por defecto que borra la mejor columna
#
# Vale la pena mirar un caso concreto de cómo se pierde una ventaja sin darse
# cuenta. `HistGradientBoostingRegressor` trae `min_samples_leaf = 20`. Es un
# valor sensato para las decenas de miles de filas con que se suele usar boosting.
#
# Aquí hay unas 115 filas de entrenamiento y la columna `promo` vale 1 en
# aproximadamente el 10 % de ellas: unas 12 semanas. Una división del árbol que
# aísle las semanas de promoción produce una hoja de 12 observaciones, y con
# `min_samples_leaf = 20` esa división **está prohibida**. El modelo recibe la
# columna más informativa de la tabla y no puede usarla nunca.
#
# El síntoma es silencioso: no hay error, no hay advertencia, el modelo ajusta y
# predice. Simplemente da el mismo resultado con y sin la columna.

# %%
filas = []
for h in range(1, H_MAX + 1):
    X = construir_features(g, h, con_calendario=True)
    utiles = X.notna().all(axis=1)
    for corte in CORTES:
        i_obj = corte + h - 1
        entrena = utiles & (X.index <= corte - 1)
        mdl = Pipeline([
            ("pre", preprocesador(X.columns)),
            ("mdl", HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.1, max_depth=3,
                min_samples_leaf=20, random_state=0))])      # <- el default
        mdl.fit(X[entrena], ys[entrena])
        filas.append({"y": y[i_obj],
                      "yhat": float(mdl.predict(X.iloc[[i_obj]])[0])})

hoja = pd.DataFrame(filas)
w_5 = res.loc["HistGB (con calendario)", "WAPE"]      # ya calculado en la sección 4
w_def = wape(hoja.y, hoja.yhat)
print("HistGB con covariables de calendario:\n")
print(f"  min_samples_leaf =  5 (el usado aquí)   WAPE = {w_5:.4f}")
print(f"  min_samples_leaf = 20 (por defecto)     WAPE = {w_def:.4f}")
print(f"\n  El valor por defecto cuesta {w_def - w_5:.4f} de WAPE, "
      f"un {w_def / w_5 - 1:.0%} más de error,")
print("  y deja al modelo por debajo de Holt-Winters aunque tenga la información")
print("  que Holt-Winters no tiene. El default no era neutral: era una decisión.")

# %% [markdown]
# ## 6. Las cuatro vistas del experimento

# %%
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5))

# --- (a) la estructura de las particiones
a = ax[0, 0]
n_demo = 60
for i, (etiqueta, cv_obj, y0) in enumerate(
        [("KFold(shuffle=True)  — entrena con el futuro",
          KFold(5, shuffle=True, random_state=0), 6.5),
         ("TimeSeriesSplit  — la prueba siempre después",
          TimeSeriesSplit(5), 0.0)]):
    for k, (tr, te) in enumerate(cv_obj.split(np.arange(n_demo))):
        fila = y0 + k
        a.scatter(tr, np.full(len(tr), fila), s=10, marker="s",
                  color="#90a4ae", label="entrenamiento" if (i + k) == 0 else None)
        a.scatter(te, np.full(len(te), fila), s=10, marker="s",
                  color="#c62828", label="prueba" if (i + k) == 0 else None)
    a.text(-1, y0 + 4.9, etiqueta, fontsize=9.5, fontweight="bold",
           color="#37474f")
a.axhline(5.7, color="#37474f", lw=0.8)
a.set_yticks([])
a.set_xlabel("semana (orden cronológico)")
a.set_title("(a) Cada fila es una partición; rojo = se evalúa ahí", loc="left")
a.legend(fontsize=8, loc="lower right", ncol=2)
a.set_ylim(-1, 12)
a.grid(False)

# --- (b) lo que estima cada esquema contra el error real
b = ax[0, 1]
nombres = list(cv.index)
px = np.arange(len(nombres))
h1 = ev[(ev.h == 1) & (ev.familia == "con calendario")]
real = [float(np.abs(h1[h1.modelo == f"{k} (con calendario)"].eval("y - yhat")).mean())
        for k in nombres]
b.bar(px - 0.3, cv["MAE KFold aleatorio"], width=0.19, color="#c62828",
      label="KFold aleatorio (inválido)")
b.bar(px - 0.1, cv["MAE TimeSeriesSplit"], width=0.19, color="#1565c0",
      label="TimeSeriesSplit(5)")
b.bar(px + 0.1, cv["MAE TSS (test=6)"], width=0.19, color="#7cb0de",
      label="TimeSeriesSplit(test=6)")
b.bar(px + 0.3, real, width=0.19, color="#37474f",
      label="error real, 12 orígenes, h = 1")
b.set_xticks(px)
b.set_xticklabels(nombres, fontsize=9)
b.set_ylabel("MAE (cajas)")
b.set_title("(b) Solo la barra oscura es el error que se va a observar", loc="left")
b.legend(fontsize=7.5)

# --- (c) WAPE por horizonte
c_ax = ax[1, 0]
curvas = [("ingenuo estacional", "#9e9e9e", ":"),
          ("Holt-Winters mult.", "#1565c0", "-"),
          ("Ridge (sin calendario)", "#f9a825", "--"),
          ("RandomForest (sin calendario)", "#6a1b9a", "--"),
          ("Ridge (con calendario)", "#c62828", "-"),
          ("HistGB (con calendario)", "#2e7d32", "-")]
for nombre, col, ls in curvas:
    d = ev[ev.modelo == nombre]
    puntos = [wape(dd.y, dd.yhat) for _, dd in d.groupby("h")]
    c_ax.plot(range(1, H_MAX + 1), puntos, "o-", color=col, ls=ls, lw=1.8, ms=5,
              label=nombre)
c_ax.set_xticks(range(1, H_MAX + 1))
c_ax.set_xlabel("horizonte h (semanas)")
c_ax.set_ylabel("WAPE")
c_ax.set_title("(c) Las líneas punteadas —sin calendario— van todas arriba",
               loc="left")
c_ax.legend(fontsize=7.5, ncol=2)

# --- (d) coeficientes de Ridge con calendario
d_ax = ax[1, 1]
Xf = construir_features(g, h=1, con_calendario=True)
ut = Xf.notna().all(axis=1) & (Xf.index <= CORTES[0] - 1)
mdl = Pipeline([("pre", preprocesador(Xf.columns)), ("mdl", Ridge(alpha=1.0))])
mdl.fit(Xf[ut], ys[ut])
nombres_col = ([c for c in Xf.columns if c not in BINARIAS]
               + [c for c in BINARIAS if c in Xf.columns])
coef = pd.Series(mdl.named_steps["mdl"].coef_, index=nombres_col)
coef = coef.reindex(coef.abs().sort_values().index)[-12:]
colores = ["#c62828" if k in BINARIAS else "#37474f" for k in coef.index]
d_ax.barh(range(len(coef)), coef.to_numpy(), color=colores)
d_ax.set_yticks(range(len(coef)))
d_ax.set_yticklabels(coef.index, fontsize=8)
d_ax.axvline(0, color="#37474f", lw=0.8)
d_ax.set_xlabel("coeficiente (variables continuas estandarizadas)")
d_ax.set_title("(d) Ridge, h = 1: en rojo las columnas del calendario", loc="left")

figura(fig, "regresion_sklearn")
plt.show()

# %% [markdown]
# El panel (a) es el que conviene mirar más rato. Las cinco filas de arriba son
# `KFold` aleatorio: los puntos rojos —lo que se evalúa— están repartidos por toda
# la serie, y para cada uno de ellos hay puntos grises **a la derecha**, es decir,
# semanas posteriores que están en el conjunto de entrenamiento. Las cinco filas
# de abajo son `TimeSeriesSplit`: el bloque rojo está siempre después del gris, y
# el gris crece. Esa segunda estructura es la que reproduce lo que pasa en la
# planta, donde el lunes se pronostica con lo que se sabe el lunes.
#
# El panel (d) cierra el argumento del cuaderno con el coeficiente más grande de
# la regresión: `promo`. No es un rezago, no es un término de Fourier, no es nada
# que se pueda extraer de la historia de la serie. Es una columna que alguien del
# área comercial tiene en una planilla.

# %% [markdown]
# ## 7. Comprobación contra las cifras de referencia

# %%
print("Comprobación:")
ok = [
    verificar(base, 0.2858, "WAPE del ingenuo estacional (la base)", tol=5e-3),
    verificar(hw, 0.1951, "WAPE de Holt-Winters mult.", tol=5e-3),
    verificar(sin_cal.min(), 0.2317, "mejor sklearn SIN calendario", tol=2e-2),
    verificar(con_cal.min(), 0.1490, "mejor sklearn CON calendario", tol=2e-2),
    verificar(res.loc["Ridge (con calendario)", "WAPE"], 0.1671,
              "Ridge con calendario", tol=2e-2),
    verificar(w_def, 0.2335, "HistGB con min_samples_leaf por defecto", tol=2e-2),
    verificar(float(cv.loc["Ridge", "MAE KFold aleatorio"]), 217.4,
              "MAE que reporta el KFold aleatorio (Ridge)", tol=2e-2),
    verificar(float(cv.loc["Ridge", "MAE TimeSeriesSplit"]), 391.6,
              "MAE que reporta TimeSeriesSplit (Ridge)", tol=2e-2),
]
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

assert (cv["MAE KFold aleatorio"] < cv["MAE TimeSeriesSplit"]).all(), \
    "el KFold aleatorio debe dar un error menor en los tres modelos"
assert sin_cal.min() > hw, \
    "a igualdad de información, ningún modelo de sklearn debía batir a Holt-Winters"
assert res.loc["Ridge (con calendario)", "WAPE"] < \
    res.loc["RandomForest (con calendario)", "WAPE"], \
    "Ridge debía batir a RandomForest: la ventaja es la información, no el algoritmo"
print("\nA igualdad de información gana el método clásico.")
print("Con covariables de calendario gana sklearn, y el mejor lineal le gana al")
print("mejor no lineal: la ventaja es la INFORMACIÓN, no el ALGORITMO.")

tabla(res.reset_index(), "sklearn_comparacion")
tabla(cv.reset_index(), "sklearn_validacion_cruzada")
resumen({"base_ingenuo_estacional": base, "holt_winters": hw,
         "mejor_sin_calendario": float(sin_cal.min()),
         "mejor_con_calendario": float(con_cal.min()),
         "histgb_default": w_def, "histgb_leaf5": w_5,
         "subestimacion_kfold": {k: float(v)
                                 for k, v in cv["subestimación"].items()}},
        "resumen_sklearn")
print("\nGuardado en resultados/")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **La validación cruzada aleatoria no es una aproximación aceptable: es un
#    número sin significado.** Entrena con el futuro, y como las filas vecinas
#    comparten rezagos, el modelo reconoce en la prueba valores que ya vio como
#    objetivo. Subestima el error entre 35 % y 45 % y —lo peor— no lo hace por
#    igual en todos los modelos, así que además cambia el ranking. `TimeSeriesSplit`
#    corrige la lógica, pero su nivel depende del diseño de los pliegues: con 5
#    particiones sobre 115 filas, la primera entrena con 20 semanas.
# 2. **La tabla `X → y` hay que armarla por horizonte.** Con la estrategia
#    directa, el modelo de horizonte *h* solo ve rezagos de *h* o más. Usar
#    `y_{t-1}` para pronosticar a cuatro semanas es una fuga de información que no
#    produce ningún error visible, solo un WAPE demasiado bueno.
# 3. **A igualdad de información, los métodos clásicos ganan en esta serie.** Con
#    182 semanas, estacionalidad fuerte y regular, y un solo SKU sin información
#    cruzada, la estructura de Holt-Winters vale más que la flexibilidad de un
#    bosque. El aprendizaje automático gana cuando hay muchas series, covariables
#    ricas y relaciones no lineales; aquí no se da ninguna de las tres.
# 4. **Separe siempre el aporte del algoritmo del de la información.** Aquí la
#    mejora vino de cuatro columnas del calendario, no del método: `Ridge`, una
#    regresión lineal, le gana a `RandomForest` cuando las dos las tienen.
#    Conseguir el plan promocional de comercial rinde más que cambiar de modelo.
#
# ## Ejercicios
#
# 1. Cambie `K` de 3 a 1 y a 8 en `construir_features` y vuelva a correr la
#    comparación con covariables. ¿Con cuántos pares de Fourier se obtiene el
#    mínimo WAPE? Compare el número de columnas con las 51 indicadoras que haría
#    falta usar para representar la misma estacionalidad, y estime cuántas
#    observaciones por parámetro quedarían en cada caso.
# 2. Rehaga la sección 4 sobre POR-3000 —la serie grumosa del cuaderno 02— en vez
#    de DUR-480. ¿Los modelos de `sklearn` siguen perdiendo contra el mejor
#    clásico de esa serie, que era SBA? Revise qué predice `RandomForest` en las
#    semanas de demanda cero y explique por qué ningún modelo de regresión
#    entrenado con error cuadrático va a pronosticar cero.
# 3. Implemente la estrategia **recursiva**: ajuste un solo modelo de horizonte 1
#    y realimente su propio pronóstico para llegar a h = 4, reemplazando el rezago
#    que falta por el valor pronosticado. Compare su WAPE por horizonte contra el
#    de la estrategia directa del panel (c). ¿En qué horizonte se separan las dos
#    curvas y por qué la diferencia crece con h?
