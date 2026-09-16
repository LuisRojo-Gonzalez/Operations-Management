# %% [markdown]
# # Unidad 2 · Métodos clásicos de pronóstico, escritos a mano
#
# **Planta:** Conservas del Itata S.A. — DUR-480 (serie suave) y POR-3000 (grumosa)
# **Datos:** `DatosClases/2_Pronostico/itata_demanda.csv`; el cuaderno 01 dejó la
# serie limpia y la clasificación ADI/CV² en `resultados/`
#
# ---
#
# ## Qué se decide aquí
#
# Dos cosas, y la segunda importa más que la primera.
#
# **Primera: qué método usar.** No se decide por gusto ni por lo que se enseñó
# más recientemente. Se decide por el cuadrante de la serie: una serie suave se
# pronostica con suavizamiento exponencial, una serie grumosa con Croston o SBA.
# Aplicar Holt-Winters a POR-3000 no es una elección discutible; es un error de
# tipo, y aquí se va a ver cuánto cuesta.
#
# **Segunda: con qué métrica se compara.** Aquí es donde se pierde la mayoría de
# los proyectos de pronóstico. Si la métrica es MAPE y la serie tiene ceros, el
# ranking que sale está mal de raíz, y todo lo que se construya encima —la
# elección del modelo, el stock de seguridad, el plan— hereda el error.
#
# Todos los métodos se escriben **a mano**, con la recursión a la vista. No es
# purismo: la recursión *es* el contenido. Ver que `l = α·y + (1−α)·l` en tres
# líneas explica en el acto por qué un α alto sigue al último dato y uno bajo lo
# promedia todo; ver `ExponentialSmoothing(...).fit()` no explica nada.
#
# ## La obligación de tener base
#
# Un Holt-Winters con WAPE de 19 % no dice **nada** hasta que se sabe cuánto da
# el ingenuo estacional. Los métodos ingenuos no son un trámite previo: son el
# examen que todo modelo tiene que aprobar antes de que valga la pena discutirlo.
# En muchas plantas el ingenuo estacional *es* lo que hace la planilla que el
# modelo pretende reemplazar.
#
# ## Cifras de referencia (una partición, últimas 26 semanas)
#
# | cantidad | valor |
# |---|---|
# | base a batir en DUR-480 — ingenuo estacional | WAPE ≈ 0,20 |
# | mejor método clásico en DUR-480 | Holt-Winters multiplicativo |
# | mejor método en POR-3000 | SBA (Croston corregido) |
# | MAPE de Holt-Winters en POR-3000 | indefinido / explota: 35 % de ceros |
# | tasa de SBA vs Croston en POR-3000 | SBA = Croston × (1 − α/2) |

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comun import datos, tabla, figura, resumen, verificar, estilo

estilo()

dem = pd.read_csv(datos("2_Pronostico", "itata_demanda.csv"))

# limpieza mínima del cuaderno 01: el error de unidad de TOM-480 en t = 87
m = (dem.sku == "TOM-480") & (dem.t == 87)
for c in ("pedidos_cajas", "despachos_cajas"):
    dem.loc[m, c] = dem.loc[m, c] / 24

M = 52                                    # período estacional: semanas por año


def serie(sku, col="pedidos_cajas"):
    g = dem[dem.sku == sku].sort_values("t")
    return g[col].to_numpy(dtype=float)


y_suave = serie("DUR-480")                # serie suave
y_grumosa = serie("POR-3000")             # serie grumosa

print(f"DUR-480 : n = {len(y_suave)}   media = {y_suave.mean():,.0f} cajas   "
      f"ceros = {(y_suave == 0).mean():.1%}")
print(f"POR-3000: n = {len(y_grumosa)}   media = {y_grumosa.mean():,.0f} cajas   "
      f"ceros = {(y_grumosa == 0).mean():.1%}")

# %% [markdown]
# ## 1. Una interfaz común, para poder comparar
#
# Todos los métodos comparten dos operaciones: `ajustar(y)` y `predecir(h)`. Con
# eso, el código de evaluación los trata a todos igual y agregar un método nuevo
# no obliga a tocar nada más. Es una decisión de diseño pequeña que se paga sola
# en el cuaderno 04, cuando haya que reajustar diez métodos en doce orígenes.
#
# La separación `ajustar` / `predecir` también deja explícito algo que se olvida:
# el ajuste solo puede ver el pasado. Un método que en `predecir` consultara
# valores posteriores al corte estaría haciendo trampa, y con esta interfaz eso
# se ve, porque `predecir` no recibe datos.

# %%
class Metodo:
    nombre = "base"

    def ajustar(self, y):
        self.y = np.asarray(y, dtype=float)
        return self

    def predecir(self, h: int):
        raise NotImplementedError

    def __repr__(self):
        return self.nombre

# %% [markdown]
# ## 2. Los métodos de referencia
#
# **Ingenuo:** repite el último valor. Es el punto de partida de todo y, en
# series financieras, es sorprendentemente difícil de batir.
#
# **Ingenuo estacional:** repite el mismo período del ciclo anterior. Con
# demanda estacional, *esta* es la base a batir. Es además el denominador de
# MASE, así que su error define la unidad de medida de toda la comparación.
#
# **Media móvil de k períodos:** promedia las últimas k observaciones. Su
# problema con datos estacionales es estructural: promedia semanas que están en
# fases distintas del ciclo, así que arrastra el pronóstico hacia el promedio
# anual y llega siempre tarde a los peaks.

# %%
class Ingenuo(Metodo):
    """Repite el último valor observado."""
    nombre = "ingenuo"

    def predecir(self, h):
        return np.repeat(self.y[-1], h)


class IngenuoEstacional(Metodo):
    """Repite el mismo período del ciclo anterior. LA base con datos estacionales."""
    nombre = "ingenuo estacional"

    def __init__(self, m=M):
        self.m = m

    def predecir(self, h):
        ult = self.y[-self.m:]                    # el último ciclo completo
        return np.array([ult[i % self.m] for i in range(h)])


class MediaMovil(Metodo):
    """Promedio de las últimas k observaciones."""

    def __init__(self, k=4):
        self.k = k
        self.nombre = f"media móvil({k})"

    def predecir(self, h):
        return np.repeat(self.y[-self.k:].mean(), h)


# demostración de la recursión estacional en tres puntos
mdl = IngenuoEstacional(M).ajustar(y_suave[:130])
print("Ingenuo estacional ajustado con y[0:130], pronóstico de h = 1..4:")
print(f"  yhat = {np.round(mdl.predecir(4), 1)}")
print(f"  que es exactamente y[{130-52}:{134-52}] = {y_suave[78:82]}")

# %% [markdown]
# ## 3. Suavizamiento exponencial simple (SES)
#
# La recursión completa cabe en una línea:
#
# $$\ell_t = \alpha\, y_t + (1-\alpha)\, \ell_{t-1}$$
#
# y el pronóstico es plano: $\hat y_{t+h} = \ell_t$ para todo *h*.
#
# Desenrollando la recursión se ve qué significa α: el peso de la observación de
# hace *k* períodos es α(1−α)^k, es decir, los pesos decaen geométricamente. Con
# α = 0,3 la observación de hace 10 semanas pesa 0,3 · 0,7¹⁰ ≈ 0,008; con
# α = 0,05 pesa 0,05 · 0,95¹⁰ ≈ 0,030, casi cuatro veces más. **α es la memoria
# del método**, y elegirlo es elegir cuánta historia se considera vigente.
#
# SES no tiene tendencia ni estacionalidad. Sobre una serie con ciclo anual eso
# se ve de inmediato: pronostica una línea horizontal a la altura del nivel
# reciente, y esa línea está por definición equivocada en el peak y en el valle.

# %%
class SES(Metodo):
    """Suavizamiento exponencial simple: l_t = a·y_t + (1-a)·l_{t-1}."""

    def __init__(self, alfa=0.3):
        self.alfa = alfa
        self.nombre = f"SES(a={alfa})"

    def ajustar(self, y):
        super().ajustar(y)
        l = self.y[0]                                  # inicialización: y_1
        for v in self.y[1:]:
            l = self.alfa * v + (1 - self.alfa) * l    # <- la recursión, completa
        self.nivel = l
        return self

    def predecir(self, h):
        return np.repeat(self.nivel, h)


print("Pesos implícitos de SES sobre las últimas observaciones:")
print(f"  {'k (semanas atrás)':<22}" + "".join(f"{k:>9}" for k in [0, 1, 2, 5, 10, 20]))
for a in (0.05, 0.30, 0.70):
    pesos = [a * (1 - a) ** k for k in [0, 1, 2, 5, 10, 20]]
    print(f"  alfa = {a:<16.2f}" + "".join(f"{p:>9.4f}" for p in pesos))
print("\n  alfa alto = memoria corta, sigue al último dato y al ruido con él.")
print("  alfa bajo = memoria larga, estable pero lento para reaccionar.")

# %% [markdown]
# ## 4. Holt: nivel más tendencia
#
# Holt agrega una segunda ecuación, que suaviza la **pendiente**:
#
# $$\ell_t = \alpha\, y_t + (1-\alpha)(\ell_{t-1} + \phi\, b_{t-1})$$
# $$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)\,\phi\, b_{t-1}$$
# $$\hat y_{t+h} = \ell_t + (\phi + \phi^2 + \dots + \phi^h)\, b_t$$
#
# El parámetro φ es el amortiguamiento y es más importante de lo que parece. Con
# φ = 1 la tendencia es lineal y se extrapola **para siempre**: si la pendiente
# estimada es +12 cajas por semana, a 52 semanas el método pronostica +624 cajas,
# aunque esa pendiente se haya estimado con ruido. Con φ = 0,9 la suma
# geométrica converge a φ/(1−φ) = 9, así que la tendencia aporta como máximo 9
# pendientes y el pronóstico se aplana.
#
# En la práctica, la tendencia amortiguada gana casi siempre en horizontes de más
# de unas pocas semanas. Es uno de los resultados más replicados de la
# literatura de pronóstico y una de las recomendaciones que más se ignoran.

# %%
class Holt(Metodo):
    """Nivel + tendencia (lineal con phi = 1, amortiguada con phi < 1)."""

    def __init__(self, alfa=0.3, beta=0.1, phi=1.0):
        self.alfa, self.beta, self.phi = alfa, beta, phi
        self.nombre = (f"Holt(a={alfa},b={beta}"
                       + (f",phi={phi})" if phi < 1 else ")"))

    def ajustar(self, y):
        super().ajustar(y)
        l = self.y[0]
        b = (self.y[1] - self.y[0]) if len(self.y) > 1 else 0.0
        for v in self.y[1:]:
            l_ant = l
            l = self.alfa * v + (1 - self.alfa) * (l + self.phi * b)
            b = self.beta * (l - l_ant) + (1 - self.beta) * self.phi * b
        self.nivel, self.pendiente = l, b
        return self

    def predecir(self, h):
        return np.array([self.nivel
                         + sum(self.phi ** k for k in range(1, i + 1)) * self.pendiente
                         for i in range(1, h + 1)])


print(f"{'phi':>6}{'pendiente estimada':>22}{'aporte a h=4':>15}{'aporte a h=52':>16}")
for phi in (1.0, 0.95, 0.90, 0.80):
    mdl = Holt(0.3, 0.05, phi).ajustar(y_suave)
    a4 = sum(phi ** k for k in range(1, 5)) * mdl.pendiente
    a52 = sum(phi ** k for k in range(1, 53)) * mdl.pendiente
    print(f"{phi:>6.2f}{mdl.pendiente:>22.2f}{a4:>15.1f}{a52:>16.1f}")
print("\n  Con phi = 1 la tendencia aporta 52 pendientes a un año vista.")
print("  Con phi = 0,8 el aporte se satura en phi/(1-phi) = 4 pendientes.")

# %% [markdown]
# ## 5. Holt-Winters: nivel, tendencia y estacionalidad
#
# La tercera ecuación suaviza los **índices estacionales**, uno por cada una de
# las m posiciones del ciclo:
#
# $$\ell_t = \alpha\,\frac{y_t}{s_{t-m}} + (1-\alpha)(\ell_{t-1} + \phi b_{t-1})
# \qquad\text{(versión multiplicativa)}$$
# $$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)\phi b_{t-1}$$
# $$s_t = \gamma\,\frac{y_t}{\ell_t} + (1-\gamma)\, s_{t-m}$$
#
# Dos decisiones aparecen aquí y las dos se toman mirando el gráfico, no por
# costumbre:
#
# **Multiplicativo o aditivo.** Multiplicativo cuando la amplitud estacional
# crece con el nivel —lo habitual en demanda: un peak de «+30 %» es más cajas
# cuando el negocio es más grande—; aditivo cuando la amplitud es constante en
# unidades. La versión multiplicativa **no está definida con ceros**, lo que la
# descarta de entrada para POR-3000.
#
# **Cuánta historia hace falta.** Con m = 52 hay 52 índices estacionales que
# estimar. Con dos ciclos completos —104 semanas— cada índice se estima con dos
# observaciones. Eso ya es poco; con menos de dos ciclos no hay nada que estimar
# y el método se niega a correr. Ajustar Holt-Winters semanal con 18 meses de
# historia es ajustar ruido y llamarlo estacionalidad.

# %%
class HoltWinters(Metodo):
    """Holt-Winters aditivo o multiplicativo con m períodos por ciclo."""

    def __init__(self, alfa=0.2, beta=0.05, gamma=0.2, m=M,
                 estacionalidad="multiplicativa", phi=1.0):
        self.alfa, self.beta, self.gamma, self.m = alfa, beta, gamma, m
        self.tipo, self.phi = estacionalidad, phi
        self.nombre = f"Holt-Winters({estacionalidad[:4]}.)"

    def ajustar(self, y):
        super().ajustar(y)
        y, m = self.y, self.m
        if len(y) < 2 * m:
            raise ValueError(f"Holt-Winters con m={m} necesita {2*m} "
                             f"observaciones y hay {len(y)}")
        # --- inicialización: promedio de cada ciclo e índices estacionales
        n_ciclos = len(y) // m
        prom = np.array([y[i * m:(i + 1) * m].mean() for i in range(n_ciclos)])
        l = prom[0]
        b = (prom[-1] - prom[0]) / max(1, (n_ciclos - 1) * m)
        if self.tipo.startswith("mult"):
            s = np.array([np.mean([y[c * m + i] / prom[c] for c in range(n_ciclos)
                                   if prom[c] > 0]) for i in range(m)])
            s = s / s.mean()                      # los índices promedian 1
        else:
            s = np.array([np.mean([y[c * m + i] - prom[c] for c in range(n_ciclos)])
                          for i in range(m)])
            s = s - s.mean()                      # los índices promedian 0
        s = list(s)
        # --- las tres recursiones
        for t, v in enumerate(y):
            st = s[t % m] if t < m else s[-m]
            l_ant = l
            if self.tipo.startswith("mult"):
                st_seg = st if abs(st) > 1e-9 else 1.0
                l = self.alfa * (v / st_seg) + (1 - self.alfa) * (l + self.phi * b)
                s.append(self.gamma * (v / l if l else 1.0) + (1 - self.gamma) * st)
            else:
                l = self.alfa * (v - st) + (1 - self.alfa) * (l + self.phi * b)
                s.append(self.gamma * (v - l) + (1 - self.gamma) * st)
            b = self.beta * (l - l_ant) + (1 - self.beta) * self.phi * b
        self.nivel, self.pendiente, self.estacional = l, b, s
        return self

    def predecir(self, h):
        out = []
        for i in range(1, h + 1):
            amort = sum(self.phi ** k for k in range(1, i + 1))
            st = self.estacional[-self.m + ((i - 1) % self.m)]
            base = self.nivel + amort * self.pendiente
            out.append(base * st if self.tipo.startswith("mult") else base + st)
        return np.array(out)


hw = HoltWinters(0.2, 0.02, 0.2, M).ajustar(y_suave)
idx = np.array(hw.estacional[-M:])
iso = dem[dem.sku == "DUR-480"].sort_values("t").iso_semana.to_numpy()
orden = np.argsort(idx)
print(f"DUR-480 — nivel = {hw.nivel:,.0f} cajas, pendiente = {hw.pendiente:+.2f}/sem")
print(f"Índices estacionales: mínimo {idx.min():.3f}, máximo {idx.max():.3f} "
      f"(amplitud {idx.max()/idx.min()-1:.0%})")
print(f"  las 3 posiciones más altas del ciclo: {[int(iso[-M:][i]) for i in orden[-3:]]}")
print(f"  las 3 más bajas                     : {[int(iso[-M:][i]) for i in orden[:3]]}")
print("  (el generador puso el peak de DUR en la semana ISO 50: fin de año)")

try:
    HoltWinters(m=M).ajustar(y_suave[:90])
except ValueError as e:
    print(f"\nCon 90 semanas: {e}")

# %% [markdown]
# ## 6. Croston y SBA: cuando la serie tiene ceros
#
# En POR-3000 el 35 % de las semanas no hay pedido. Aplicarle SES a esa serie
# produce un número que no corresponde a ninguna semana real: ni a las de cero
# ni a las de pedido. Es el promedio de dos poblaciones distintas presentado como
# si fuera una.
#
# Croston (1972) hace lo que hay que hacer: **separa las dos preguntas**.
# Suaviza por un lado el tamaño del pedido cuando hay pedido (z) y por otro el
# intervalo entre pedidos (q), cada uno con su propia recursión, y actualiza
# **solo cuando ocurre una demanda**. La tasa esperada es el cociente:
#
# $$\hat y = \frac{z_t}{q_t}$$
#
# Ese estimador está sesgado al alza, y el sesgo no es pequeño. Syntetos y Boylan
# (2005) mostraron que multiplicar por (1 − α/2) corrige la mayor parte; el
# método resultante se llama SBA. Con α = 0,1 la corrección es del 5 %. Suena
# poco hasta que se recuerda que el stock de seguridad de un SKU intermitente es
# del mismo orden de magnitud.
#
# Nótese lo que Croston **no** hace: no dice cuándo llegará el próximo pedido.
# Da una tasa por período, plana. Para un SKU institucional eso es exactamente lo
# que se necesita para dimensionar inventario, y exactamente lo que no sirve para
# programar un camión.

# %%
class Croston(Metodo):
    """Croston (1972) y su corrección de sesgo SBA (Syntetos-Boylan 2005)."""

    def __init__(self, alfa=0.1, sba=False):
        self.alfa, self.sba = alfa, sba
        self.nombre = ("SBA" if sba else "Croston") + f"(a={alfa})"

    def ajustar(self, y):
        super().ajustar(y)
        z = q = None                  # tamaño suavizado, intervalo suavizado
        intervalo = 1
        for v in self.y:
            if v > 0:
                if z is None:                        # primera demanda observada
                    z, q = float(v), float(intervalo)
                else:
                    z = self.alfa * v + (1 - self.alfa) * z
                    q = self.alfa * intervalo + (1 - self.alfa) * q
                intervalo = 1                        # se reinicia el contador
            else:
                intervalo += 1                       # NO se actualiza z ni q
        if z is None:
            self.tasa = 0.0
        else:
            self.z, self.q = z, q
            self.tasa = z / q
            if self.sba:
                self.tasa *= (1 - self.alfa / 2.0)
        return self

    def predecir(self, h):
        return np.repeat(self.tasa, h)


cro = Croston(0.1).ajustar(y_grumosa)
sba = Croston(0.1, sba=True).ajustar(y_grumosa)
ses_g = SES(0.3).ajustar(y_grumosa)

print("POR-3000, ajustado sobre las 182 semanas:")
print(f"  tamaño medio suavizado z   = {cro.z:,.1f} cajas por pedido")
print(f"  intervalo medio suavizado q= {cro.q:.3f} semanas entre pedidos")
print(f"  tasa Croston = z/q         = {cro.tasa:,.2f} cajas/semana")
print(f"  tasa SBA = Croston x (1-a/2)= {sba.tasa:,.2f} cajas/semana "
      f"({sba.tasa/cro.tasa-1:+.1%})")
print(f"  a modo de contraste, SES(0.3) = {ses_g.nivel:,.2f} cajas/semana")
print(f"  media empírica de la serie    = {y_grumosa.mean():,.2f} cajas/semana")
print(f"  media de las semanas con pedido = "
      f"{y_grumosa[y_grumosa > 0].mean():,.2f} cajas")

# %% [markdown]
# ## 7. Las métricas: por qué MAPE no sirve aquí
#
# MAPE = media de |y − ŷ| / y. Tiene tres defectos, y los tres se activan en
# estos datos:
#
# 1. **No está definida con ceros.** El 35 % de las semanas de POR-3000 tienen
#    y = 0 y la división explota. La salida habitual —excluir esos períodos— es
#    peor que el problema: se evalúa el método justamente en las semanas que le
#    resultan fáciles y se ignoran aquellas en que pronosticar 78 cajas donde
#    hubo 0 fue un error completo.
# 2. **Es asimétrica.** Sobreestimar tiene error porcentual sin cota (pronosticar
#    300 donde hubo 100 da 200 %) mientras que subestimar tiene error acotado por
#    100 % (pronosticar 0 donde hubo 100 da 100 %). Minimizar MAPE **premia
#    pronósticos sesgados a la baja**, que en inventario significa quiebre.
# 3. **No pondera por volumen.** Equivocarse en 100 cajas de un SKU de 10.000
#    pesa lo mismo que en uno de 200.
#
# Las dos métricas que sí sirven:
#
# **WAPE** = Σ|y − ŷ| / Σ|y|. Agrega el error en unidades y lo divide por el
# volumen total. Está bien definida con ceros mientras el total no sea cero, y es
# la métrica natural cuando lo que importa es el inventario. Es la que se usa en
# todo el resto de la unidad.
#
# **MASE** = MAE / MAE del ingenuo estacional **en entrenamiento**. Es
# adimensional y comparable entre SKU, que es lo que MAPE promete y no cumple. Y
# tiene una lectura inmediata: **MASE < 1 significa mejor que repetir el año
# pasado; MASE > 1 significa que el modelo no justifica su existencia**. El
# denominador se calcula en entrenamiento a propósito: escalarlo con la muestra
# de prueba haría que el denominador dependiera de lo que se quiere predecir.

# %%
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def wape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    s = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / s) if s else float("nan")


def mape(y, yhat):
    """Solo sobre los períodos con y != 0. Con ceros no está definida."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])))


def mase(y, yhat, y_entrenamiento, m=M):
    """Escalada por el error del ingenuo estacional EN ENTRENAMIENTO."""
    ytr = np.asarray(y_entrenamiento, float)
    if len(ytr) <= m:
        m = 1
    escala = np.mean(np.abs(ytr[m:] - ytr[:-m]))
    return float(mae(y, yhat) / escala) if escala else float("nan")


def sesgo_relativo(y, yhat):
    """Positivo = se subestimó la demanda."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sum(y - yhat) / np.sum(np.abs(y))) if np.sum(np.abs(y)) else 0.0


# la asimetría de MAPE, en un ejemplo de dos observaciones
print("La asimetría de MAPE — demanda real y = 100 en ambos casos:")
for yhat_v, etiqueta in [(50, "subestimo a la mitad"), (200, "sobreestimo al doble")]:
    print(f"  ŷ = {yhat_v:>3}  ({etiqueta:<22}) -> "
          f"MAPE = {mape([100], [yhat_v]):>6.0%}   WAPE = {wape([100], [yhat_v]):>6.0%}")
print("  El mismo error absoluto (50 cajas) recibe MAPE 50 % o 100 % según el")
print("  signo. Minimizar MAPE es pedirle al modelo que se quede corto.")

# %% [markdown]
# ## 8. Comparación sobre una partición: DUR-480, la serie suave
#
# Se entrena con todo salvo las últimas 26 semanas y se pronostica ese medio año.
# Es una evaluación **provisional**: una sola partición da un solo número con
# varianza enorme, y si el tramo de prueba cae en el peak cualquier método parece
# malo. El cuaderno 04 lo hace bien con origen rodante. Aquí sirve para ver el
# orden de magnitud y para dejar claro cuál es la base.

# %%
H = 26
corte = len(y_suave) - H
entrena, prueba = y_suave[:corte], y_suave[corte:]

metodos_suave = {
    "ingenuo": Ingenuo(),
    "ingenuo estacional": IngenuoEstacional(M),
    "media móvil(4)": MediaMovil(4),
    "media móvil(13)": MediaMovil(13),
    "SES(a=0.3)": SES(0.3),
    "Holt(phi=1)": Holt(0.3, 0.05, 1.0),
    "Holt(phi=0.9)": Holt(0.3, 0.05, 0.9),
    "Holt-Winters mult.": HoltWinters(0.2, 0.02, 0.2, M, "multiplicativa"),
    "Holt-Winters adit.": HoltWinters(0.2, 0.02, 0.2, M, "aditiva"),
}

filas, pred_suave = [], {}
for nombre, mdl in metodos_suave.items():
    yhat = mdl.ajustar(entrena).predecir(H)
    pred_suave[nombre] = yhat
    filas.append({"método": nombre, "WAPE": wape(prueba, yhat),
                  "MASE": mase(prueba, yhat, entrena), "MAE": mae(prueba, yhat),
                  "RMSE": rmse(prueba, yhat), "MAPE": mape(prueba, yhat),
                  "sesgo rel.": sesgo_relativo(prueba, yhat)})

res_suave = pd.DataFrame(filas).set_index("método").sort_values("WAPE")
print("DUR-480 — entrenamiento t = 1..156, prueba t = 157..182")
print(res_suave.round(4).to_string())

base_s = res_suave.loc["ingenuo estacional", "WAPE"]
mejor_s = res_suave.index[0]
print(f"\nBase (ingenuo estacional): WAPE = {base_s:.4f}")
print(f"Mejor método: {mejor_s} -> mejora {1 - res_suave.WAPE.iloc[0]/base_s:.1%}")
peores = [k for k in res_suave.index if res_suave.loc[k, "WAPE"] > base_s]
print(f"NO superan la base: {', '.join(peores) if peores else 'ninguno'}")

# %% [markdown]
# Dos lecturas que conviene no saltarse.
#
# **Holt-Winters multiplicativo gana, y gana por la estacionalidad, no por la
# tendencia.** La comparación que lo prueba es Holt contra Holt-Winters: misma
# maquinaria de nivel y tendencia, la única diferencia es la tercera ecuación.
# La caída del WAPE entre ambos es el valor de modelar el ciclo anual.
#
# **La media móvil de 13 semanas es peor que la de 4.** Promediar un trimestre
# completo de una serie estacional mezcla semanas de fases distintas del ciclo y
# arrastra el pronóstico hacia el promedio anual. Más suavizamiento no es más
# estabilidad cuando lo que se suaviza es señal.

# %% [markdown]
# ## 9. Comparación sobre POR-3000, la serie grumosa
#
# Aquí aparecen dos cosas a la vez: qué método corresponde, y cómo la métrica
# equivocada da el ranking equivocado.

# %%
corte_g = len(y_grumosa) - H
entrena_g, prueba_g = y_grumosa[:corte_g], y_grumosa[corte_g:]

metodos_grumosa = {
    "ingenuo": Ingenuo(),
    "ingenuo estacional": IngenuoEstacional(M),
    "media móvil(4)": MediaMovil(4),
    "SES(a=0.3)": SES(0.3),
    "Holt(phi=0.9)": Holt(0.3, 0.05, 0.9),
    "Holt-Winters adit.": HoltWinters(0.2, 0.02, 0.2, M, "aditiva"),
    "Croston(a=0.1)": Croston(0.1),
    "SBA(a=0.1)": Croston(0.1, sba=True),
    "SBA(a=0.2)": Croston(0.2, sba=True),
}

filas, pred_grumosa = [], {}
for nombre, mdl in metodos_grumosa.items():
    yhat = mdl.ajustar(entrena_g).predecir(H)
    pred_grumosa[nombre] = yhat
    filas.append({"método": nombre, "WAPE": wape(prueba_g, yhat),
                  "MASE": mase(prueba_g, yhat, entrena_g),
                  "MAPE (solo y>0)": mape(prueba_g, yhat),
                  "sesgo rel.": sesgo_relativo(prueba_g, yhat)})

res_grumosa = pd.DataFrame(filas).set_index("método")
print("POR-3000 — ordenado por WAPE (la métrica correcta):")
print(res_grumosa.sort_values("WAPE").round(4).to_string())
print("\nEl MISMO conjunto de resultados, ordenado por MAPE:")
print(res_grumosa.sort_values("MAPE (solo y>0)").round(4).to_string())

n_cero = int((prueba_g == 0).sum())
print(f"\nEn la ventana de prueba hay {n_cero} semanas de {H} con demanda cero.")
print(f"MAPE las descarta: se calcula sobre {H - n_cero} observaciones, no {H}.")
print(f"Ranking por WAPE : {list(res_grumosa.sort_values('WAPE').index[:3])}")
print(f"Ranking por MAPE : {list(res_grumosa.sort_values('MAPE (solo y>0)').index[:3])}")

# %% [markdown]
# Los dos rankings no coinciden, y la razón es exactamente la que se anticipó:
# MAPE evalúa a cada método solo en las semanas en que hubo pedido —las fáciles—
# y le perdona gratis todo lo que pronosticó en las semanas de cero. Un método
# que pronostique alto queda bien en MAPE porque nunca se le mide el exceso donde
# la demanda fue nula.
#
# Un comentario aparte sobre la magnitud del WAPE. En POR-3000 todos los métodos
# están en torno a 0,8: el error es del orden del 80 % del volumen. **Eso no es
# un fracaso del analista**, es la naturaleza de una serie grumosa: la demanda es
# un proceso puntual y ningún método plano puede acertarle a una serie que salta
# entre 0 y 400. Lo honesto es reportarlo, decirle a la planta que ese SKU se
# gestiona por inventario y no por pronóstico, y dimensionar el stock de
# seguridad con la σ del error —que es lo que hace el cuaderno 04— en vez de
# prometer una precisión que no existe.

# %% [markdown]
# ## 10. Los pronósticos, vistos

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# --- (a) DUR-480
a = ax[0]
t = np.arange(1, len(y_suave) + 1)
a.plot(t[-78:], y_suave[-78:], lw=1.4, color="#37474f", label="demanda real")
a.axvline(corte + 0.5, color="#1565c0", lw=1.5, ls=":")
tp = t[corte:]
for nombre, col in [("ingenuo estacional", "#9e9e9e"), ("SES(a=0.3)", "#f9a825"),
                    ("Holt(phi=0.9)", "#6a1b9a"), ("Holt-Winters mult.", "#c62828")]:
    a.plot(tp, pred_suave[nombre], lw=1.8, color=col,
           label=f"{nombre} (WAPE {res_suave.loc[nombre,'WAPE']:.3f})")
a.text(corte + 2, a.get_ylim()[1] * 0.97, "prueba", color="#1565c0", fontsize=9,
       va="top")
a.set_title("(a) DUR-480 — serie suave: la estacionalidad es lo que hay que modelar",
            loc="left")
a.set_ylabel("cajas")
a.legend(fontsize=8, ncol=2, loc="lower left")

# --- (b) POR-3000
b = ax[1]
tg = np.arange(1, len(y_grumosa) + 1)
b.bar(tg[-78:], y_grumosa[-78:], width=0.85, color="#b0bec5",
      label="demanda real (barras)")
b.axvline(corte_g + 0.5, color="#1565c0", lw=1.5, ls=":")
tpg = tg[corte_g:]
for nombre, col in [("SES(a=0.3)", "#f9a825"), ("Croston(a=0.1)", "#1565c0"),
                    ("SBA(a=0.1)", "#c62828"),
                    ("Holt-Winters adit.", "#6a1b9a")]:
    b.plot(tpg, pred_grumosa[nombre], lw=1.8, color=col,
           label=f"{nombre} (WAPE {res_grumosa.loc[nombre,'WAPE']:.3f})")
b.set_title("(b) POR-3000 — serie grumosa: ningún método plano le acierta a un "
            "proceso puntual", loc="left")
b.set_xlabel("semana t")
b.set_ylabel("cajas")
b.legend(fontsize=8, ncol=2)

figura(fig, "metodos_clasicos")
plt.show()

# %% [markdown]
# El panel (b) deja ver de un golpe por qué el WAPE de POR-3000 no puede bajar de
# 0,7: las líneas de pronóstico son necesariamente planas y las barras van de 0 a
# más de 400. Lo único que se puede pedirle a un método aquí es que acierte la
# **tasa media** —el área bajo la curva— y que no tenga sesgo. Croston y SBA
# hacen precisamente eso; Holt-Winters aditivo intenta además ponerle una forma
# estacional a un proceso que no la tiene, y empeora.

# %% [markdown]
# ## 11. Comprobación contra las cifras de referencia

# %%
print("Comprobación:")
ok = [
    verificar(sba.tasa / cro.tasa, 1 - 0.1 / 2, "SBA / Croston = (1 - alfa/2)"),
    verificar(res_suave.loc["ingenuo estacional", "MASE"], 1.0,
              "MASE del ingenuo estacional (≈ 1 por construcción)", tol=0.35),
    verificar(float(np.ptp(pred_suave["SES(a=0.3)"])), 0.0,
              "SES pronostica una línea plana"),
    verificar(float(np.ptp(pred_suave["ingenuo"])), 0.0,
              "el ingenuo pronostica una línea plana"),
    verificar(res_suave.WAPE.min(), base_s * 0.85,
              "el mejor método mejora al menos 15 % sobre la base", tol=0.25),
]
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

assert mejor_s.startswith("Holt-Winters"), \
    f"en una serie suave y estacional debía ganar Holt-Winters, ganó {mejor_s}"
mejor_g = res_grumosa.WAPE.idxmin()
assert mejor_g.startswith(("SBA", "Croston")), \
    f"en la serie grumosa debía ganar Croston o SBA, ganó {mejor_g}"
assert res_grumosa.loc["Holt-Winters adit.", "WAPE"] > res_grumosa.loc["SBA(a=0.1)", "WAPE"], \
    "Holt-Winters no debería batir a SBA en una serie grumosa"
print(f"\nSerie suave   -> gana {mejor_s}")
print(f"Serie grumosa -> gana {mejor_g}")
print("El cuadrante ADI/CV² del cuaderno 01 predijo los dos resultados.")

tabla(res_suave.reset_index(), "clasicos_dur480")
tabla(res_grumosa.reset_index(), "clasicos_por3000")
resumen({"mejor_suave": mejor_s, "wape_mejor_suave": float(res_suave.WAPE.min()),
         "base_suave": float(base_s), "mejor_grumosa": mejor_g,
         "wape_mejor_grumosa": float(res_grumosa.WAPE.min()),
         "tasa_croston": float(cro.tasa), "tasa_sba": float(sba.tasa)},
        "resumen_clasicos")
print("\nGuardado en resultados/")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **El cuadrante decide el método.** DUR-480 es suave y gana Holt-Winters;
#    POR-3000 es grumosa y gana SBA. No hubo que probar: el diagnóstico del
#    cuaderno 01 ya lo había dicho. Probar sirvió para confirmarlo y para medir
#    cuánto cuesta equivocarse.
# 2. **Sin base no hay resultado.** Un WAPE de 0,19 es excelente o mediocre según
#    cuánto dé el ingenuo estacional. Reportar el error de un modelo sin el de la
#    base es reportar la mitad de la información, y es la mitad que no permite
#    decidir.
# 3. **MAPE da el ranking equivocado cuando hay ceros.** No está definida, es
#    asimétrica —premia quedarse corto, que en inventario es quiebre— y no pondera
#    por volumen. WAPE agrega en unidades; MASE compara contra la base y se lee
#    solo: menos de 1 es mejor que repetir el año pasado.
# 4. **Amortiguar la tendencia casi siempre paga.** Con φ = 1 el método extrapola
#    para siempre una pendiente estimada con ruido; con φ < 1 el aporte se satura
#    en φ/(1−φ) pendientes. Es de los resultados más replicados de la literatura
#    y de los más ignorados en la práctica.
#
# ## Ejercicios
#
# 1. Barra α de 0,05 a 0,95 en `SES` sobre DUR-480 y grafique el WAPE de prueba
#    contra α. ¿Dónde está el mínimo? Repita con los datos de POR-3000. ¿Por qué
#    el α óptimo es tan distinto entre las dos series, y qué dice eso sobre dejar
#    α fijo en 0,3 para todo el catálogo?
# 2. Implemente el método de **deriva** (último valor más la pendiente media de
#    toda la historia, ŷ_{t+h} = y_t + h·(y_t − y_1)/(n−1)) y agréguelo a las dos
#    comparaciones. ¿En cuál de las dos series es competitivo y por qué? Compárelo
#    con `Holt(phi=1)` y explique en qué se parecen.
# 3. Ajuste `HoltWinters` sobre DUR-480 usando solo las últimas 104 semanas y
#    compare su WAPE de prueba con el del ajuste sobre las 156. ¿Mejora o empeora?
#    Ahora hágalo con 130 semanas. Con lo que observe, escriba en dos frases la
#    regla que le daría a la planta sobre cuánta historia usar.
