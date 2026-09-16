# %% [markdown]
# # Unidad 3 · CLSP: lot-sizing multiproducto con capacidad
#
# **Planta:** Conservas del Itata S.A.
# **Datos:** `DatosClases/3_Planificacion/Planificiacion/` — 12 semanas, 6 SKU, 4 recursos
# **Solver:** Gurobi
#
# ---
#
# ## Qué se decide aquí
#
# En el cuaderno anterior cada SKU se resolvía por separado y no había
# capacidad. Los seis problemas eran independientes y Wagner-Whitin los cerraba
# en milisegundos. Aquí se agrega **una sola cosa** —que las máquinas son
# finitas y compartidas— y el problema cambia de naturaleza.
#
# La planta tiene cuatro recursos:
#
# * **PREP**, las marmitas donde se cocina el preparado a granel;
# * **L1**, la línea de envasado del formato 480 g;
# * **L2**, la línea de los formatos 820 g y 3000 g;
# * **AUT**, el autoclave de esterilización, por donde pasa **todo**.
#
# Las horas disponibles no son constantes: hay mantenciones programadas y un
# feriado que recortan semanas concretas. Y la demanda tampoco: la semana 3 es
# la de Fiestas Patrias y concentra un 40 % más que una semana normal.
#
# ## Lo que se rompe al agregar capacidad
#
# 1. **Se cae la propiedad de Wagner-Whitin.** Sin capacidad, en el óptimo
#    nunca se produce en una semana que arrastra inventario (`I_{t-1}·x_t = 0`).
#    Con capacidad eso es falso: si la semana 3 no cabe en horas, hay que
#    *adelantar* producción, y entonces el inventario de entrada y la
#    producción son positivos a la vez. Sin esa propiedad, la recursión de la
#    programación dinámica no tiene de dónde agarrarse.
# 2. **Los SKU dejan de ser independientes.** Comparten AUT y comparten línea.
#    Ya no hay seis problemas de 12 semanas: hay uno con 72 binarias acopladas.
# 3. **El CLSP es NP-difícil** incluso con un solo recurso. No hay algoritmo
#    polinómico conocido. Queda el MIP, y con el MIP vuelve a importar el
#    **big-M**.
#
# ## El big-M en presencia de capacidad
#
# El cuaderno 01 usó `M = demanda restante desde t`. Aquí aparece una **segunda
# cota**, independiente y muchas veces más fuerte: lo que *físicamente cabe* en
# el recurso más restrictivo de esa semana,
#
# $$M_{it} = \min\Big(\underbrace{\textstyle\sum_{u \ge t} d_{iu}}_{\text{cota económica}},\;
#   \underbrace{\min_{r}\frac{\text{horas}_{rt} - s_{ir}}{a_{ir}}}_{\text{cota física}}\Big)$$
#
# Las dos son válidas; la que sirve es el mínimo. Y la segunda depende de `t`,
# así que el big-M **no es un número: es una matriz** (i, t).
#
# ## Cifras de referencia
#
# | resultado | valor |
# |---|---|
# | CLSP básico, objetivo | **16 457 114 CLP** |
# | preparaciones | **34** |
# | gap de la relajación lineal en la raíz, big-M ajustado | **54,2 %** |
# | gap en la raíz, big-M = 10⁶ | **99,2 %** |
# | recurso cuello de botella | **AUT** (92 % de sus horas regulares) |
# | pares (i, t) que violan la propiedad de Wagner-Whitin | **7** |
#
# **Sobre el objetivo.** El costo variable de producción (13 400 CLP por caja
# de DUR-480, etc.) está **excluido**. No hay pérdida de venta ni rutas
# alternativas, así que ese término es prácticamente una constante: se lleva
# el 98 % del gasto total y no depende de las decisiones que el modelo toma.
# Dejarlo dentro esconde el intercambio que sí importa. Se reporta aparte.

# %%
import sys
import time
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

from comun import datos, tabla, figura, resumen, verificar, estilo

estilo()

dem = pd.read_csv(datos("3_Planificacion", "itata_demanda_plan.csv"))
par = pd.read_csv(datos("3_Planificacion", "itata_parametros_sku.csv"))
cap = pd.read_csv(datos("3_Planificacion", "itata_capacidad.csv"))
tas = pd.read_csv(datos("3_Planificacion", "itata_tasas.csv"))
rec = pd.read_csv(datos("3_Planificacion", "itata_recursos.csv"))
ini = pd.read_csv(datos("3_Planificacion", "itata_inventario_inicial.csv"))

SKUS = list(par.sku)
T = sorted(dem.t.unique().tolist())
REC = sorted(cap.recurso.unique().tolist())

print(f"{len(SKUS)} SKU · {len(T)} semanas · {len(REC)} recursos")
print(rec.to_string(index=False))

# %% [markdown]
# ## 1. Los datos, en las estructuras que el modelo va a usar
#
# Tres cosas que conviene mirar antes de escribir una sola restricción.
#
# **El consumo es por recurso y por caja.** Cada SKU pasa por PREP (cocción del
# granel), por **su** línea de envasado (L1 o L2, no ambas) y por AUT. El
# archivo `itata_tasas.csv` trae las tres tasas en horas por caja.
#
# **El tiempo de setup se consume en la línea**, no en todos los recursos. Es
# el CIP de la llenadora: cambiar de producto obliga a limpiar esa máquina. Por
# eso `tiempo_setup` está indexado en (SKU, su línea).
#
# **La capacidad varía por semana.** `itata_capacidad.csv` no repite las horas
# nominales de `itata_recursos.csv`: les descuenta las mantenciones. Ignorar esa
# columna es el error más caro del cuaderno, porque produce un plan que no cabe
# justo en las semanas en que la planta está más apretada.
#
# Un detalle de pandas que muerde: la columna del archivo de inventario se
# llama `item`, y `fila.item` en pandas **no** devuelve esa columna sino el
# método `Series.item`. Hay que escribir `df["item"]`. El síntoma es un
# inventario inicial silenciosamente igual a cero.

# %%
demanda = {(r.sku, int(r.t)): float(r.demanda_cajas) for _, r in dem.iterrows()}
p = par.set_index("sku")
costo_setup = {i: float(p.loc[i, "costo_setup_clp"]) for i in SKUS}
costo_almacen = {i: float(p.loc[i, "costo_almacen_clp_caja_sem"]) for i in SKUS}
costo_prod = {i: float(p.loc[i, "costo_unitario_clp_caja"]) for i in SKUS}
stock_seg = {i: float(p.loc[i, "stock_seguridad_cajas"]) for i in SKUS}

# OJO: df["item"], no df.item  (ver la nota de arriba)
_ini = dict(zip(ini["item"], ini.inventario_inicial.astype(float)))
inv_ini = {i: _ini.get(i, 0.0) for i in SKUS}

linea = dict(zip(tas.sku, tas.linea))

# consumo[(sku, recurso)] = horas por caja
consumo = {}
for _, r in tas.iterrows():
    consumo[(r.sku, "PREP")] = float(r.h_por_caja_PREP)
    consumo[(r.sku, "AUT")] = float(r.h_por_caja_AUT)
    consumo[(r.sku, r.linea)] = float(r.h_por_caja_linea)

# el setup se paga en HORAS sobre la linea de envasado del SKU
tiempo_setup = {(r.sku, linea[r.sku]): float(r.tiempo_setup_h) for _, r in par.iterrows()}

capacidad = {(r.recurso, int(r.t)): float(r.horas_disponibles) for _, r in cap.iterrows()}
extra_max = {(r.recurso, int(r.t)): float(r.horas_extra_max) for _, r in cap.iterrows()}

print("Inventario inicial (cajas):", {i: inv_ini[i] for i in SKUS})
print("\nSemanas con capacidad reducida:\n")
print(cap[cap.observacion.notna()][["t", "recurso", "horas_disponibles",
                                    "observacion"]].to_string(index=False))

# %% [markdown]
# ## 2. ¿Cabe la demanda en la planta?
#
# Antes de optimizar hay que contestar una pregunta que la optimización no
# contesta bien: **¿dónde está el cuello de botella y en qué semanas aprieta?**
#
# La carga de un recurso `r` en la semana `t`, si se produjera exactamente la
# demanda de esa semana, es
#
# $$\text{carga}_{rt} = \sum_i a_{ir}\, d_{it} \;+\; \sum_i s_{ir}$$
#
# El segundo término supone que se prepara todo (seis setups). Es una cota
# pesimista, pero es la relevante: la semana apretada es justo aquella en que
# no se puede dejar de preparar nada.

# %%
filas = []
for t in T:
    fila = {"t": t, "demanda_cajas": sum(demanda[(i, t)] for i in SKUS)}
    for r in REC:
        carga = sum(consumo.get((i, r), 0.0) * demanda[(i, t)] for i in SKUS)
        carga += sum(tiempo_setup.get((i, r), 0.0) for i in SKUS)
        fila[f"{r}_%"] = 100.0 * carga / capacidad[(r, t)]
    filas.append(fila)
util = pd.DataFrame(filas)
print("Utilizacion si la demanda de cada semana se produjera en esa semana (%):\n")
print(util.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

print("\nCarga del horizonte completo (solo produccion, sin setups):\n")
carga_horizonte = {}
for r in REC:
    carga = sum(consumo.get((i, r), 0.0) * demanda[(i, t)] for i in SKUS for t in T)
    reg = sum(capacidad[(r, t)] for t in T)
    ext = sum(extra_max[(r, t)] for t in T)
    carga_horizonte[r] = 100 * carga / reg
    print(f"  {r:5s}  carga {carga:7.1f} h | regular {reg:6.1f} h ({100*carga/reg:5.1f} %) "
          f"| con extra {reg+ext:6.1f} h ({100*carga/(reg+ext):5.1f} %)")

cuello_previo = max(REC, key=lambda r: carga_horizonte[r])
print(f"\nCuello de botella del horizonte: {cuello_previo}")

# %% [markdown]
# ### La lectura
#
# **AUT es el cuello de botella del horizonte.** Ocupa el 92 % de sus horas
# regulares solo con la producción de la demanda, sin contar setups ni
# desbalances. Es el único recurso por el que pasan los seis SKU: L1 y L2 se
# reparten los productos; PREP y AUT los ven a todos.
#
# **La semana 3 no cabe.** Tres de los cuatro recursos pasan del 100 %: AUT,
# L1 y PREP. Es la semana de Fiestas Patrias y la demanda salta a 8 465 cajas
# contra las ~6 200 de una semana normal. Esa demanda **no se puede producir en
# la semana 3**: hay que adelantarla, y adelantarla cuesta inventario.
#
# Ese solo hecho predice la forma del plan óptimo antes de resolverlo, y ya
# destruye la propiedad de Wagner-Whitin: en las semanas 1 y 2 habrá producción
# **y** inventario de arrastre al mismo tiempo, que es exactamente lo que sin
# capacidad nunca conviene.
#
# Nótese también la semana 6: AUT al 105 % por la mantención programada del
# autoclave, con demanda perfectamente normal. El cuello de botella no siempre
# lo crea la demanda; a veces lo crea el calendario de mantención.

# %% [markdown]
# ## 3. El big-M ajustado por (i, t)
#
# La restricción de enlace `x_it ≤ M_it · y_it` dice "no se produce sin
# preparar". El valor de `M_it` no cambia el óptimo entero mientras sea válido
# —es decir, mientras no recorte ninguna solución factible—, pero cambia
# radicalmente el costo de encontrarlo.
#
# Dos cotas válidas, por razones distintas:
#
# * **Económica.** Nunca conviene producir en `t` más que la demanda que queda
#   por atender de `t` en adelante, más las reservas que el plan deba dejar en
#   bodega. Producir de más solo agrega inventario que nadie consume.
#   *(Esta cota **deja de ser válida si se permite backlog**; se vuelve sobre
#   ello en el cuaderno 03, y la función ya lleva el interruptor.)*
# * **Física.** Aunque conviniera, no cabe: descontadas las horas de setup, el
#   recurso más restrictivo de esa semana limita cuántas cajas se pueden
#   correr. Esta cota es la que aporta la capacidad y no existía en el
#   cuaderno 01.
#
# El mínimo de las dos es válido y es el que se usa.
#
# **Una decisión de diseño que hay que declarar.** Los cuadernos 02, 03 y 04
# comparten esta función, y varias de las variantes del cuaderno 03 admiten
# horas extra y obligan a dejar stock de seguridad. Para que la MISMA cota sea
# válida en toda la familia, el término físico cuenta también las horas extra
# disponibles y el término económico suma las reservas. Eso la vuelve más
# floja que una cota hecha a medida de este modelo básico. La sección 6
# cuantifica exactamente cuánto cuesta esa comodidad.

# %%
# el modelo debe cerrar el horizonte con al menos el stock de seguridad en
# bodega; si no, vacia la bodega en t = 12 y el plan no es implementable
inv_final_min = dict(stock_seg)


def big_M(i, t, con_backlog=False, con_extra=True, con_reserva=True):
    """Cota superior VALIDA y ajustada para la produccion de i en t."""
    if con_backlog:
        # con backlog, demanda de CUALQUIER periodo puede seguir pendiente
        resto = sum(demanda[(i, u)] for u in T) - inv_ini[i]
    else:
        resto = sum(demanda[(i, u)] for u in T if u >= t)
    resto = max(0.0, resto)
    if con_reserva:
        resto += stock_seg[i] + inv_final_min[i]

    fisico = float("inf")
    for r in REC:
        a = consumo.get((i, r), 0.0)
        if a <= 0:
            continue
        horas = capacidad.get((r, t), 0.0) - tiempo_setup.get((i, r), 0.0)
        if con_extra:
            horas += extra_max.get((r, t), 0.0)
        fisico = min(fisico, max(0.0, horas) / a)
    return float(min(resto, fisico))


filas = []
for i in SKUS:
    for t in (1, 3, 12):
        eco = sum(demanda[(i, u)] for u in T if u >= t) + stock_seg[i] + inv_final_min[i]
        fis = min((capacidad[(r, t)] + extra_max[(r, t)] - tiempo_setup.get((i, r), 0.0))
                  / consumo[(i, r)] for r in REC if consumo.get((i, r), 0.0) > 0)
        filas.append({"sku": i, "t": t, "cota_economica": eco, "cota_fisica": fis,
                      "M_it": big_M(i, t), "manda": "fisica" if fis < eco else "economica"})
cotas = pd.DataFrame(filas)
print(cotas.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
print("\nEn los SKU de alto volumen manda la cota fisica; en los de bajo volumen")
print("y al final del horizonte, la economica. Quedarse con una sola de las dos")
print("deja la mitad del efecto sobre la mesa.")

# %% [markdown]
# ## 4. El modelo
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i,t} K_i\, y_{it} \;+\; \sum_{i,t} h_i\, I_{it} \\
# \text{s.a.}\quad & I_{i,t-1} + x_{it} - d_{it} = I_{it} && \forall i, t \\
# & x_{it} \le M_{it}\, y_{it} && \forall i, t \\
# & \sum_i \big( a_{ir}\, x_{it} + s_{ir}\, y_{it} \big) \le \text{cap}_{rt} && \forall r, t \\
# & x, I \ge 0,\quad y \in \{0,1\}
# \end{aligned}
# $$
#
# Lo único nuevo respecto del cuaderno 01 es la tercera línea, y es la que
# acopla los seis SKU. Nótese que el setup consume **horas** además de dinero:
# `s_ir · y_it` está dentro de la restricción de capacidad. Un modelo que cobra
# el setup solo en el objetivo produce planes que no caben en la planta.

# %%
def clsp(big_M_generico=None, big_M_a_medida=False, relajado=False,
         tiempo_limite=120.0, gap=1e-4, verbose=False):
    """CLSP basico multiproducto con capacidad. Devuelve (modelo, variables)."""
    m = gp.Model("CLSP")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = tiempo_limite
    m.Params.MIPGap = gap
    m.Params.Seed = 0

    vt = GRB.CONTINUOUS if relajado else GRB.BINARY
    x = m.addVars(SKUS, T, lb=0.0, name="x")                     # cajas producidas
    y = m.addVars(SKUS, T, vtype=vt, lb=0.0, ub=1.0, name="y")   # hubo preparacion
    I = m.addVars(SKUS, T, lb=0.0, name="I")                     # inventario al cierre

    # --- balance de inventario
    for i in SKUS:
        for k, t in enumerate(T):
            previo = inv_ini[i] if k == 0 else I[i, T[k - 1]]
            m.addConstr(previo + x[i, t] - demanda[(i, t)] == I[i, t],
                        name=f"bal[{i},{t}]")

    # --- enlace produccion - preparacion
    M = {}
    for i in SKUS:
        for t in T:
            if big_M_generico:
                M[i, t] = big_M_generico
            elif big_M_a_medida:
                # cota valida SOLO para este modelo: sin horas extra, sin reservas
                M[i, t] = big_M(i, t, con_extra=False, con_reserva=False)
            else:
                M[i, t] = big_M(i, t)
            m.addConstr(x[i, t] <= M[i, t] * y[i, t], name=f"enl[{i},{t}]")

    # --- capacidad: la produccion Y el setup consumen horas
    for r in REC:
        for t in T:
            m.addConstr(
                gp.quicksum(consumo.get((i, r), 0.0) * x[i, t]
                            + tiempo_setup.get((i, r), 0.0) * y[i, t] for i in SKUS)
                <= capacidad[(r, t)], name=f"cap[{r},{t}]")

    c_setup = gp.quicksum(costo_setup[i] * y[i, t] for i in SKUS for t in T)
    c_alm = gp.quicksum(costo_almacen[i] * I[i, t] for i in SKUS for t in T)
    m.setObjective(c_setup + c_alm, GRB.MINIMIZE)

    m.optimize()
    return m, {"x": x, "y": y, "I": I, "M": M,
               "c_setup": c_setup, "c_almacen": c_alm}


ESTADO = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
          GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD"}

t0 = time.perf_counter()
m, v = clsp()
seg = time.perf_counter() - t0

n_setups = int(round(sum(v["y"][i, t].X for i in SKUS for t in T)))
costo_variable = sum(costo_prod[i] * v["x"][i, t].X for i in SKUS for t in T)

print(f"estado          : {ESTADO.get(m.Status, m.Status)}")
print(f"objetivo        : {m.ObjVal:,.0f} CLP")
print(f"cota inferior   : {m.ObjBound:,.0f} CLP      gap final {m.MIPGap:.4%}")
print(f"variables       : {m.NumVars}  ({m.NumBinVars} binarias)")
print(f"restricciones   : {m.NumConstrs}")
print(f"tiempo          : {seg:.2f} s      nodos explorados {int(m.NodeCount)}")
print()
print(f"  costo de setup          : {v['c_setup'].getValue():>13,.0f} CLP  "
      f"({n_setups} preparaciones)")
print(f"  costo de almacenamiento : {v['c_almacen'].getValue():>13,.0f} CLP")
print(f"  [excluido] costo variable de produccion: {costo_variable:>13,.0f} CLP")
print(f"  ese termino es el {100*costo_variable/(costo_variable+m.ObjVal):.1f} % del gasto total")
print("  y es casi constante: por eso se reporta aparte y no se optimiza.")

# %% [markdown]
# ## 5. La propiedad de Wagner-Whitin, comprobada y refutada
#
# Sin capacidad, en el óptimo `I_{i,t-1} · x_{it} = 0` para todo (i, t): cada
# lote cubre un bloque contiguo de semanas y nunca se produce sobre inventario
# existente. Es la propiedad que hace funcionar la programación dinámica.
#
# Con capacidad la pregunta es empírica: ¿la cumple el óptimo del CLSP?

# %%
viol = []
for i in SKUS:
    for k, t in enumerate(T):
        if k == 0:
            continue
        inv_entrada = v["I"][i, T[k - 1]].X
        prod = v["x"][i, t].X
        if inv_entrada > 1e-6 and prod > 1e-6:
            viol.append({"sku": i, "t": t, "inv_entrada": inv_entrada,
                         "produccion": prod})
vio = pd.DataFrame(viol)
print(f"Pares (i, t) con produccion e inventario de entrada positivos: {len(vio)}\n")
print(vio.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
print("\nCada fila es un contraejemplo a la propiedad de Wagner-Whitin. La")
print("programacion dinamica del cuaderno 01 no puede representar estos planes:")
print("su recursion supone que cada lote arranca con la bodega vacia. No es que")
print("la PD sea lenta aqui: es que no sirve.")

# %% [markdown]
# ## 6. El precio de un big-M perezoso
#
# `M = 10⁶` es válido: nadie va a producir un millón de cajas de conserva en
# una semana. Da **el mismo óptimo entero**. Lo que cambia es la cota con que
# el solver arranca:
#
# $$\text{gap en la raíz} = \frac{z_{\text{entero}} - z_{\text{LP}}}{z_{\text{entero}}}$$
#
# Se comparan tres versiones de la misma formulación:
#
# 1. **genérico**, `M = 10⁶`;
# 2. **ajustado (familia)**, el de la sección 3, válido también con horas extra
#    y con reservas, que es el que usan los cuadernos 03 y 04;
# 3. **ajustado a medida**, válido **solo** para este modelo básico: sin horas
#    extra en la cota física y sin reservas en la económica.

# %%
filas = []
for etiqueta, kw in [("generico = 1e6", dict(big_M_generico=1e6)),
                     ("ajustado (familia)", dict()),
                     ("ajustado a medida", dict(big_M_a_medida=True))]:
    mr, _ = clsp(relajado=True, **kw)
    z_lp = mr.ObjVal
    t0 = time.perf_counter()
    me, ve = clsp(**kw)
    seg_e = time.perf_counter() - t0
    filas.append({"big-M": etiqueta, "z_LP": z_lp, "z_entero": me.ObjVal,
                  "gap_raiz_%": 100 * (me.ObjVal - z_lp) / me.ObjVal,
                  "nodos": int(me.NodeCount), "segundos": seg_e,
                  "M_medio": np.mean(list(ve["M"].values()))})
bm = pd.DataFrame(filas)
print(bm.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\nMismo optimo entero en las tres, cota inferior muy distinta.")
print("El big-M no es un detalle de implementacion: es parte de la formulacion.")
print("Y la version 'de familia' paga ~9 puntos de gap por ser reutilizable en")
print("modelos que aqui no se resuelven. Ese precio hay que saber que se paga.")

# %% [markdown]
# ## 7. El plan y el perfil de utilización
#
# El gráfico que hay que mirar: horas usadas contra horas disponibles, recurso
# por recurso y semana por semana. Es la vista que un jefe de planta reconoce,
# y es donde se ve que el modelo llenó el autoclave hasta el tope en las
# semanas previas para poder sobrevivir la semana 3.

# %%
uso = {}
for r in REC:
    for t in T:
        uso[(r, t)] = sum(consumo.get((i, r), 0.0) * v["x"][i, t].X
                          + tiempo_setup.get((i, r), 0.0) * round(v["y"][i, t].X)
                          for i in SKUS)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex=True)
for ax, r in zip(axes.ravel(), REC):
    horas = [capacidad[(r, t)] for t in T]
    usado = [uso[(r, t)] for t in T]
    sat = [t for k, t in enumerate(T) if usado[k] >= horas[k] - 1e-6]
    ax.bar(T, usado, color="#4c72b0", width=0.62, label="horas usadas")
    ax.step([t - 0.5 for t in T] + [T[-1] + 0.5], horas + [horas[-1]],
            where="post", color="#c44e52", lw=2.0, label="capacidad regular")
    if sat:
        ax.plot(sat, [uso[(r, t)] for t in sat], ls="none", marker="v",
                color="#c44e52", ms=8, zorder=5, label="saturado")
    ax.set_title(f"{r}   (utilización media {100*np.mean(usado)/np.mean(horas):.0f} %)")
    ax.set_ylabel("horas")
axes[0, 0].legend(loc="lower left", fontsize=8, ncol=3)
for ax in axes[1]:
    ax.set_xlabel("semana")
    ax.set_xticks(T)
fig.suptitle("Plan óptimo del CLSP: horas usadas contra capacidad disponible",
             fontweight="bold")
figura(fig, "clsp_utilizacion")
plt.show()

cuello = max(REC, key=lambda r: sum(uso[(r, t)] for t in T) /
             sum(capacidad[(r, t)] for t in T))
saturadas = [(r, t) for r in REC for t in T
             if uso[(r, t)] >= capacidad[(r, t)] - 1e-6]
print(f"Recurso con mayor utilizacion media en el plan: {cuello}")
print(f"Pares (recurso, semana) saturados al 100 %: {len(saturadas)}")
print("  " + ", ".join(f"{r}/t{t}" for r, t in saturadas))

# %% [markdown]
# ## 8. La semana focal
#
# La semana 3 es la que explica el plan. Vale la pena mirarla sola: cuánta
# demanda tiene, cuánto se produce en ella y cuánto llegó adelantado desde las
# semanas 1 y 2.

# %%
filas = []
for i in SKUS:
    filas.append({"sku": i, "demanda_t3": demanda[(i, 3)],
                  "produccion_t3": v["x"][i, 3].X,
                  "inv_entrada_t3": v["I"][i, 2].X,
                  "setup_t3": int(round(v["y"][i, 3].X))})
foco = pd.DataFrame(filas)
foco["cubierto_adelantando_%"] = 100 * foco.inv_entrada_t3 / foco.demanda_t3
print("Semana 3 (Fiestas Patrias):\n")
print(foco.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

for r in REC:
    nec = sum(consumo.get((i, r), 0.0) * demanda[(i, 3)] for i in SKUS)
    nec += sum(tiempo_setup.get((i, r), 0.0) for i in SKUS)
    disp = capacidad[(r, 3)]
    marca = "  <-- NO CABE" if nec > disp else ""
    print(f"  {r:5s}: producir la demanda de t=3 en t=3 pide {nec:6.1f} h "
          f"y hay {disp:5.1f} h{marca}")
print("\nEse deficit es exactamente lo que el plan adelanta a las semanas 1 y 2,")
print("y es la razon por la que la propiedad de Wagner-Whitin no sobrevive.")

# %% [markdown]
# ## 9. Comprobación

# %%
print("Comprobación:")
ok = []
ok.append(verificar(m.ObjVal, 16_457_114.0, "objetivo del CLSP básico (CLP)", tol=1e-5))
ok.append(verificar(n_setups, 34, "número de preparaciones", tol=1e-9))
ok.append(verificar(float(bm.loc[1, "gap_raiz_%"]), 54.2,
                    "gap LP en la raíz, big-M ajustado (%)", tol=1e-2))
ok.append(verificar(float(bm.loc[0, "gap_raiz_%"]), 99.2,
                    "gap LP en la raíz, big-M genérico (%)", tol=1e-2))
ok.append(verificar(len(vio), 7, "violaciones de la propiedad de Wagner-Whitin",
                    tol=1e-9))
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

# factibilidad recalculada desde los datos, sin preguntarle al solver
peor = max(uso[(r, t)] - capacidad[(r, t)] for r in REC for t in T)
assert peor <= 1e-6, f"el plan viola la capacidad por {peor:.6f} h"
for i in SKUS:
    prev = inv_ini[i]
    for t in T:
        prev = prev + v["x"][i, t].X - demanda[(i, t)]
        assert abs(prev - v["I"][i, t].X) < 1e-6, f"balance roto en {i},{t}"
assert (abs(bm.z_entero - bm.z_entero.iloc[0]) < 1e-3).all(), \
    "las tres versiones del big-M deben dar el MISMO optimo entero"
assert bm.loc[0, "gap_raiz_%"] > bm.loc[1, "gap_raiz_%"] > bm.loc[2, "gap_raiz_%"], \
    "cotas mas ajustadas deben dar gaps mas chicos"
assert cuello == "AUT", "el cuello de botella deberia ser el autoclave"
print("Factibilidad recalculada desde los CSV: capacidad y balance, correctos.")

plan = pd.DataFrame([{"sku": i, "t": t, "demanda": demanda[(i, t)],
                      "produccion": v["x"][i, t].X, "inventario": v["I"][i, t].X,
                      "setup": int(round(v["y"][i, t].X))}
                     for i in SKUS for t in T])
tabla(plan, "clsp_plan")
tabla(util, "clsp_utilizacion_demanda")
tabla(bm, "clsp_bigM")
resumen({"objetivo_clsp": float(m.ObjVal), "n_setups": n_setups,
         "gap_raiz_generico_pct": float(bm.loc[0, "gap_raiz_%"]),
         "gap_raiz_ajustado_pct": float(bm.loc[1, "gap_raiz_%"]),
         "gap_raiz_a_medida_pct": float(bm.loc[2, "gap_raiz_%"]),
         "costo_variable_excluido": float(costo_variable),
         "cuello_de_botella": cuello,
         "n_violaciones_WW": int(len(vio))}, "resumen_clsp")
print("Guardado en resultados/")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **La capacidad no agrega una restricción: cambia el problema.** Se cae la
#    propiedad de Wagner-Whitin —hay 7 pares (i, t) que la violan en el óptimo—
#    y con ella se cae la programación dinámica. El CLSP es NP-difícil.
# 2. **El big-M con capacidad tiene dos cotas, no una.** La económica (demanda
#    restante) y la física (lo que cabe en el recurso más restrictivo de esa
#    semana). Usar el mínimo baja el gap en la raíz de 99 % a 54 %, con el mismo
#    óptimo entero; ajustarlo a medida de este modelo lo baja a 45 %.
# 3. **Una cota reutilizable es una cota más floja.** Si la misma función debe
#    servir para variantes con horas extra y con reservas, hay que aflojarla, y
#    eso se paga en gap. Es una decisión legítima, pero hay que tomarla a
#    conciencia y documentarla, no heredarla.
# 4. **El setup consume horas, no solo dinero.** Si el tiempo de setup no entra
#    en la restricción de capacidad, el plan que sale no cabe en la planta.
# 5. **Identificar el cuello de botella es trabajo previo a optimizar.** AUT
#    está al 92 % de sus horas regulares en el horizonte completo. Todo lo que
#    decide este modelo es, en el fondo, un reparto de horas de autoclave.
# 6. **El costo variable de producción no se optimiza: se reporta.** Es el 98 %
#    del gasto y es casi constante. Meterlo en el objetivo hace que cualquier
#    mejora de planificación se vea como ruido decimal.
#
# ## Ejercicios
#
# 1. Elimine la restricción de capacidad de AUT y vuelva a resolver. ¿Cuánto
#    baja el objetivo? Esa diferencia es el valor de las horas de autoclave; con
#    ella discuta si conviene la inversión en un segundo autoclave.
# 2. Resuelva dejando el tiempo de setup **fuera** de la restricción de
#    capacidad. El objetivo baja. Verifique con el cálculo de horas de la
#    sección 7 que el plan resultante no cabe: es un plan más barato y falso.
# 3. Suba la demanda de la semana 3 en pasos de 10 % y resuelva. ¿En qué punto
#    el modelo se vuelve infactible? Antes de correrlo, prediga qué recurso será
#    el culpable y compruébelo con `m.computeIIS()`.
# 4. La función `big_M` tiene el interruptor `con_backlog`, que aquí nunca se
#    usa. Lea la rama y explique, **antes** de abrir el cuaderno 03, por qué la
#    cota económica deja de ser válida cuando se permite diferir demanda.
# 5. Cuente cuántos pares (recurso, semana) quedan saturados al 100 % en el
#    plan óptimo. ¿Son los mismos que la tabla de la sección 2 anticipaba? Si no
#    lo son, explique qué hizo el modelo para descomprimirlos.
