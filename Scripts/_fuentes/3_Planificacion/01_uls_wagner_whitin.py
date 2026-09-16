# %% [markdown]
# # Unidad 3 · Lot-sizing sin capacidad: Wagner-Whitin y dos formulaciones MIP
#
# **Planta:** Conservas del Itata S.A.
# **Datos:** `DatosClases/3_Planificacion/Planificiacion/` — horizonte de 12 semanas, 6 SKU
# **Solver:** Gurobi
#
# ---
#
# ## Qué se decide aquí
#
# El problema es el más simple de la unidad y por eso sirve para ver una idea
# que después no se vuelve a ver con esta claridad: **dos modelos correctos del
# mismo problema pueden costar cosas muy distintas de resolver.**
#
# Hay una demanda conocida por semana. Producir en una semana cuesta un costo
# fijo de preparación (el CIP de la línea) independiente de cuánto se produzca.
# Guardar cuesta por caja y por semana. No hay capacidad: se puede producir todo
# lo que se quiera. La pregunta es en qué semanas producir y cuánto.
#
# El intercambio es transparente: agrupar semanas ahorra preparaciones y paga
# inventario. Lo que no es transparente es dónde está el óptimo, y ahí entran
# los modelos.
#
# ## Tres formas de resolverlo
#
# 1. **Programación dinámica (Wagner-Whitin, 1958).** Exacta y polinómica.
#    Se apoya en una propiedad estructural: en el óptimo **nunca** se produce
#    en una semana en que queda inventario de la anterior (`I_{t-1} · x_t = 0`).
#    Es decir, cada lote cubre un bloque contiguo de semanas completo.
# 2. **MIP agregado (big-M).** La formulación que todo el mundo escribe primero.
#    Correcta, y con una relajación lineal floja.
# 3. **MIP desagregado** por período de origen. El mismo problema, más
#    variables, y relajación lineal **exacta**: gap 0 % en la raíz.
#
# Que el tercero tenga gap cero no es una curiosidad. Significa que su poliedro
# describe exactamente la envoltura convexa de las soluciones enteras, así que
# el solver resuelve un LP y termina, sin explorar un solo nodo.
#
# ## Cifras de referencia
#
# | resultado | valor |
# |---|---|
# | MIP agregado vs. Wagner-Whitin | coinciden **exactamente** en los 6 SKU |
# | gap LP en la raíz, agregado | > 0 % |
# | gap LP en la raíz, desagregado | **0 %** |

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from comun import datos, tabla, resumen, verificar, estilo

estilo()

dem = pd.read_csv(datos("3_Planificacion", "itata_demanda_plan.csv"))
par = pd.read_csv(datos("3_Planificacion", "itata_parametros_sku.csv"))
inv0 = pd.read_csv(datos("3_Planificacion", "itata_inventario_inicial.csv"))

SKUS = sorted(dem.sku.unique())
T = sorted(dem.t.unique())
print(f"{len(SKUS)} SKU · {len(T)} semanas")

p = par.set_index("sku")
demanda = {(r.sku, int(r.t)): float(r.demanda_cajas) for _, r in dem.iterrows()}
setup = {s: float(p.loc[s, "costo_setup_clp"]) for s in SKUS}
almacen = {s: float(p.loc[s, "costo_almacen_clp_caja_sem"]) for s in SKUS}
i0 = {r.item: float(r.inventario_inicial) for _, r in inv0.iterrows()}

par[["sku", "costo_setup_clp", "costo_almacen_clp_caja_sem"]]

# %% [markdown]
# ## 1. Wagner-Whitin por programación dinámica
#
# La recursión aprovecha la propiedad de los bloques contiguos. Si `F(t)` es el
# costo mínimo de cubrir las semanas `1..t-1`, entonces producir en la semana
# `j` para cubrir hasta `t-1` cuesta el setup más el inventario de arrastrar esa
# demanda desde `j`:
#
# $$F(t) = \min_{1 \le j < t} \; F(j) + K + h \sum_{u=j}^{t-1} (u - j)\, d_u$$
#
# El término `(u - j)` es el número de semanas que la demanda de `u` espera en
# bodega. No hay que evaluar todas las combinaciones de semanas de producción:
# basta con probar desde qué semana arranca cada bloque.

# %%
def wagner_whitin(d, K, h, inv_inicial=0.0):
    """Devuelve (costo_optimo, lotes). d es una lista indexada desde 0."""
    n = len(d)
    d = list(d)
    # El inventario inicial se consume primero: reduce la demanda temprana.
    resto = inv_inicial
    for i in range(n):
        usar = min(resto, d[i])
        d[i] -= usar
        resto -= usar

    F = [0.0] + [float("inf")] * n
    desde = [0] * (n + 1)
    for t in range(1, n + 1):
        for j in range(1, t + 1):
            # producir en j para cubrir j..t
            inv = sum((u - j) * d[u - 1] for u in range(j, t + 1))
            c = F[j - 1] + (K if sum(d[j - 1:t]) > 1e-9 else 0.0) + h * inv
            if c < F[t] - 1e-9:
                F[t], desde[t] = c, j

    lotes, t = {}, n
    while t > 0:
        j = desde[t]
        q = sum(d[j - 1:t])
        if q > 1e-9:
            lotes[j] = q
        t = j - 1
    return F[n], lotes


filas = []
for s in SKUS:
    d = [demanda[(s, t)] for t in T]
    c, lotes = wagner_whitin(d, setup[s], almacen[s], i0.get(s, 0.0))
    filas.append({"sku": s, "costo_WW": c, "n_lotes": len(lotes),
                  "semanas_de_produccion": " ".join(str(k) for k in sorted(lotes))})
ww = pd.DataFrame(filas)
print(ww.to_string(index=False))
print(f"\nCosto total Wagner-Whitin: {ww.costo_WW.sum():,.0f} CLP")

# %% [markdown]
# ## 2. MIP agregado (big-M)
#
# La formulación directa:
#
# $$\min \sum_t K y_t + h I_t \quad\text{s.a.}\quad I_{t-1} + x_t - d_t = I_t,\;\; x_t \le M y_t$$
#
# La restricción `x_t ≤ M y_t` es el enlace: no se puede producir sin declarar
# preparación. El problema es **M**. Con `M` enorme (10⁶, por decir algo) la
# relajación lineal puede poner `y_t = x_t / M ≈ 0` y pagar casi nada de setup:
# la cota inferior queda inservible.
#
# El `M` más pequeño que sigue siendo válido es la demanda restante desde `t`:
# nunca conviene producir en `t` más de lo que queda por atender. Esa elección
# no cambia el óptimo entero —**cambia el costo de encontrarlo**.

# %%
def uls_mip(d, K, h, inv_inicial=0.0, desagregado=False, big_M=None,
            relajado=False):
    n = len(d)
    m = gp.Model("ULS")
    m.Params.OutputFlag = 0

    y = m.addVars(n, vtype=GRB.CONTINUOUS if relajado else GRB.BINARY,
                  ub=1.0, name="y")
    I = m.addVars(n, lb=0.0, name="I")

    if not desagregado:
        x = m.addVars(n, lb=0.0, name="x")
        for t in range(n):
            prev = inv_inicial if t == 0 else I[t - 1]
            m.addConstr(prev + x[t] - d[t] == I[t])
            M = big_M if big_M else max(0.0, sum(d[t:]) - (inv_inicial if t == 0 else 0))
            m.addConstr(x[t] <= M * y[t])
        obj = gp.quicksum(K * y[t] + h * I[t] for t in range(n))
    else:
        # w[t,u] = cajas producidas en t para atender la demanda de u
        w = m.addVars([(t, u) for t in range(n) for u in range(t, n)],
                      lb=0.0, name="w")
        for u in range(n):
            cubierto = inv_inicial if u == 0 else 0.0
            # el inventario inicial se reparte por orden de llegada
            m.addConstr(gp.quicksum(w[t, u] for t in range(u + 1))
                        >= d[u] - _cubre_inicial(d, inv_inicial, u))
        for t in range(n):
            for u in range(t, n):
                # cada porcion esta acotada por su propia demanda: esta es la
                # desigualdad que hace integral la formulacion
                m.addConstr(w[t, u] <= d[u] * y[t])
        obj = gp.quicksum(K * y[t] for t in range(n)) + \
              gp.quicksum(h * (u - t) * w[t, u]
                          for t in range(n) for u in range(t, n))

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        return None, None
    return m.ObjVal, m


def _cubre_inicial(d, inv, u):
    """Cuanto del inventario inicial alcanza a cubrir la demanda del periodo u."""
    resto = inv
    for i in range(u + 1):
        usar = min(resto, d[i])
        resto -= usar
        if i == u:
            return usar
    return 0.0


filas = []
for s in SKUS:
    d = [demanda[(s, t)] for t in T]
    v_ag, _ = uls_mip(d, setup[s], almacen[s], i0.get(s, 0.0))
    v_ww = float(ww[ww.sku == s].costo_WW.iloc[0])
    filas.append({"sku": s, "Wagner-Whitin": v_ww, "MIP agregado": v_ag,
                  "diferencia": v_ag - v_ww})
cmp_ = pd.DataFrame(filas)
print(cmp_.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("\nLa diferencia es cero en los 6 SKU: el MIP no es una aproximación,")
print("es el mismo problema escrito de otra forma.")

# %% [markdown]
# ## 3. Las dos formulaciones y su relajación lineal
#
# Ahora la comparación que importa. Se resuelve cada formulación **relajando la
# integralidad** de `y` (es decir, dejando `0 ≤ y ≤ 1` continuo) y se mide qué
# tan lejos queda esa cota del óptimo entero:
#
# $$\text{gap en la raíz} = \frac{z_{\text{entero}} - z_{\text{LP}}}{z_{\text{entero}}}$$
#
# Un gap grande significa que el solver arranca con una cota inferior mala y
# tiene que ramificar mucho para cerrarla.

# %%
filas = []
for s in SKUS:
    d = [demanda[(s, t)] for t in T]
    K, h, ini = setup[s], almacen[s], i0.get(s, 0.0)

    z_ent, _ = uls_mip(d, K, h, ini)
    z_lp_ag, _ = uls_mip(d, K, h, ini, relajado=True)
    z_lp_gen, _ = uls_mip(d, K, h, ini, big_M=1e6, relajado=True)
    z_lp_des, _ = uls_mip(d, K, h, ini, desagregado=True, relajado=True)
    z_ent_des, _ = uls_mip(d, K, h, ini, desagregado=True)

    filas.append({
        "sku": s, "optimo": z_ent,
        "gap_agregado_%": 100 * (z_ent - z_lp_ag) / z_ent,
        "gap_bigM_1e6_%": 100 * (z_ent - z_lp_gen) / z_ent,
        "gap_desagregado_%": 100 * (z_ent - z_lp_des) / z_ent,
        "optimo_desagregado": z_ent_des,
    })
gaps = pd.DataFrame(filas)
print(gaps.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

print(f"\n  gap medio, big-M ajustado : {gaps['gap_agregado_%'].mean():6.2f} %")
print(f"  gap medio, big-M = 1e6    : {gaps['gap_bigM_1e6_%'].mean():6.2f} %")
print(f"  gap medio, desagregado    : {gaps['gap_desagregado_%'].mean():6.2f} %")

# %% [markdown]
# ### Lo que hay que leer en esa tabla
#
# Las tres columnas de gap describen **el mismo óptimo entero**. Lo único que
# cambia es la calidad de la cota con que el solver arranca:
#
# * **big-M = 10⁶** da la peor cota. Es el valor que uno escribe cuando "quiere
#   estar seguro de que no corta nada", y el precio es un árbol de búsqueda
#   enorme.
# * **big-M ajustado** mejora mucho sin cambiar una línea del modelo: solo se
#   eligió mejor una constante.
# * **desagregado** da gap **cero**. El LP ya devuelve la solución entera.
#
# La desigualdad que hace la diferencia es `w[t,u] ≤ d_u · y_t`: acota cada
# porción por *su propia* demanda en vez de por la suma de todas. Es más
# restrictiva en la relajación y, aun así, no elimina ninguna solución entera.
#
# **Advertencia sobre el big-M ajustado.** La cota "demanda restante desde t"
# deja de ser válida si se permite diferir demanda (*backlog*), porque entonces
# parte de la demanda de semanas anteriores puede seguir pendiente y hay que
# producirla. El síntoma es venenoso: el modelo sigue diciendo `OPTIMAL` y
# devuelve un costo **mayor** que el óptimo real. No hay error ni
# infactibilidad, solo una respuesta peor. Se vuelve sobre esto en el cuaderno
# de extensiones.

# %% [markdown]
# ## 4. El plan, semana a semana

# %%
s = "TOM-480"
d = [demanda[(s, t)] for t in T]
c, lotes = wagner_whitin(d, setup[s], almacen[s], i0.get(s, 0.0))

inv, filas = i0.get(s, 0.0), []
for k, t in enumerate(T):
    q = lotes.get(k + 1, 0.0)
    inv = inv + q - d[k]
    filas.append({"t": t, "demanda": d[k], "produccion": q, "inventario": inv,
                  "setup": int(q > 1e-9)})
plan = pd.DataFrame(filas)
print(f"Plan óptimo de {s}   (costo {c:,.0f} CLP)\n")
print(plan.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
print(f"\n  preparaciones: {int(plan.setup.sum())} de {len(T)} semanas posibles")
print(f"  costo de setup: {int(plan.setup.sum()) * setup[s]:,.0f} CLP")
print(f"  costo de inventario: {almacen[s] * plan.inventario.sum():,.0f} CLP")
print("\nSe cumple la propiedad de Wagner-Whitin: en toda semana con producción,")
print("el inventario que venía de la semana anterior es cero.")

# %% [markdown]
# ## 5. Comprobación

# %%
print("Comprobación:")
ok = []
for _, r in cmp_.iterrows():
    ok.append(verificar(r["MIP agregado"], r["Wagner-Whitin"],
                        f"MIP == WW en {r['sku']}", tol=1e-6))
ok.append(verificar(gaps["gap_desagregado_%"].max(), 0.0,
                    "gap máximo del desagregado (%)", tol=1e-4))
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

assert gaps["gap_desagregado_%"].max() < 1e-4, "el desagregado debería tener gap 0"
assert (gaps["gap_bigM_1e6_%"] >= gaps["gap_agregado_%"] - 1e-9).all(), \
    "el big-M genérico no puede dar mejor cota que el ajustado"

tabla(gaps, "gaps_formulaciones")
tabla(plan, "plan_TOM480")
resumen({"costo_total_WW": float(ww.costo_WW.sum()),
         "gap_medio_agregado_pct": float(gaps["gap_agregado_%"].mean()),
         "gap_medio_bigM_1e6_pct": float(gaps["gap_bigM_1e6_%"].mean()),
         "gap_medio_desagregado_pct": float(gaps["gap_desagregado_%"].mean())},
        "resumen_uls")
print("\nGuardado en resultados/")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **Modelo correcto ≠ modelo bien formulado.** Las tres formulaciones dan el
#    mismo óptimo; una lo entrega resolviendo un LP y otra ramificando.
# 2. **El big-M es una decisión de modelado, no un número cualquiera.** El más
#    chico que sigue siendo válido es el que hay que usar, y hay que poder
#    justificar por qué es válido.
# 3. **Wagner-Whitin no es "el método viejo".** Es exacto, es rápido y su
#    propiedad estructural (`I_{t-1} · x_t = 0`) explica *por qué* los planes
#    óptimos se ven como se ven.
# 4. Todo esto ocurre **sin capacidad**. En cuanto se agrega, la propiedad de
#    los bloques contiguos se cae y la programación dinámica deja de servir:
#    ese es el cuaderno siguiente.
#
# ## Ejercicios
#
# 1. Multiplique por 10 el costo de preparación de TOM-480 y vuelva a correr.
#    ¿Cuántos lotes quedan? Explique la dirección del cambio antes de correrlo.
# 2. Verifique numéricamente la propiedad de Wagner-Whitin en los 6 SKU:
#    compruebe que en toda semana con producción el inventario de entrada es
#    cero. ¿Se cumple siempre? ¿Qué pasa con el inventario inicial?
# 3. Cuente variables y restricciones de ambas formulaciones para n = 12 y para
#    n = 52. ¿Cuál crece más rápido? Con ese dato, ¿cuándo dejaría de convenir
#    el desagregado pese a su gap cero?
