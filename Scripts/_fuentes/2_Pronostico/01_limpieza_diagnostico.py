# %% [markdown]
# # Unidad 2 · Limpieza y diagnóstico de la serie de demanda
#
# **Planta:** Conservas del Itata S.A. — 6 SKU, 3 familias, 182 semanas
# **Datos:** `DatosClases/2_Pronostico/itata_demanda.csv` (1.092 filas),
# `itata_maestro_sku.csv`, `itata_calendario.csv`, `itata_componentes_verdad.csv`
#
# ---
#
# ## Qué se decide aquí
#
# Antes de elegir un método de pronóstico hay que decidir **sobre qué serie** se
# va a pronosticar. Esa decisión se toma una sola vez, casi siempre en silencio,
# y condiciona todo lo que viene después: si se pronostica la columna
# equivocada, el mejor modelo del mundo pronostica bien la cosa equivocada.
#
# En Itata hay cuatro decisiones concretas, y las cuatro tienen consecuencias
# que se pagan en la bodega:
#
# | decisión | opción cómoda | opción correcta | qué cuesta equivocarse |
# |---|---|---|---|
# | ¿pedidos o despachos? | despachos (es lo que factura) | pedidos, o despachos descensurados | subestimación **acumulativa** del nivel |
# | un valor 24× más alto | recortarlo (winsorizar) | dividirlo por 24 | se borra una semana real o se inventa un dato |
# | una semana en cero | demanda = 0 | dato faltante | se hunde el nivel y la estacionalidad de enero |
# | un salto de nivel | promediar antes/después | razón interanual | se estima el salto con el signo cambiado |
#
# El orden tampoco es negociable: **primero se limpia, después se
# diagnostica**. En estos datos, estimar el quiebre estructural antes de
# corregir el error de unidad lo ubica en t = 108 con un salto de +48 %; el
# quiebre real está en t ≈ 121 con +33 %.
#
# ## Cifras de referencia
#
# | cantidad | valor |
# |---|---|
# | errores de unidad con el criterio correcto | 1 (TOM-480, t = 87, factor 24) |
# | detecciones del mismo criterio **sin** la guarda de intermitencia | 23 → 22 falsos positivos |
# | quiebre sobre la serie cruda **sin limpiar** | t = 86, +49,6 % (los tres números están mal) |
# | quiebre sobre la serie limpia, sin razón interanual | t = 108, +48,5 % |
# | quiebre sobre la razón interanual (m = 52) | **t = 121, +33,0 %** |
# | filas censuradas | 107 de 1.092 (9,8 %) |
# | subestimación del volumen al usar despachos | 2,67 % (excluyendo cierre de planta) |
# | clasificación SBC de POR-3000 | grumosa: ADI = 1,54 · CV² = 0,620 · 35,2 % de ceros |

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
maestro = pd.read_csv(datos("2_Pronostico", "itata_maestro_sku.csv"))
cal = pd.read_csv(datos("2_Pronostico", "itata_calendario.csv"))
verdad = pd.read_csv(datos("2_Pronostico", "itata_componentes_verdad.csv"))

print(f"{len(dem)} filas = {dem.t.nunique()} semanas x {dem.sku.nunique()} SKU")
print(f"familias: {sorted(dem.familia.unique())}")
print(f"periodo:  {dem.semana.min()} a {dem.semana.max()}")
dem.head()

# %% [markdown]
# ## 1. Censura: lo que se despachó no es lo que se pidió
#
# El archivo trae dos columnas que en una empresa real casi nunca vienen
# separadas: `pedidos_cajas` es la demanda que entró al sistema y
# `despachos_cajas` es lo que efectivamente salió de bodega. Cuando hubo
# producto, coinciden. Cuando se agotó el stock, el despacho se topa y el pedido
# sigue arriba: la demanda quedó **censurada por la derecha**, y lo que se
# registra es una cota inferior de la demanda, no la demanda.
#
# La tentación es pronosticar despachos porque es el dato que existe y el que
# cuadra con la facturación. El problema es que la censura no es simétrica:
# nunca despacha *de más*. Por lo tanto el error que introduce tiene siempre el
# mismo signo, y se acumula período tras período. Peor todavía, se retroalimenta:
# un pronóstico sesgado a la baja produce un plan de producción bajo, que produce
# más quiebres, que producen más censura, que hunde el pronóstico siguiente. Es
# una espiral que en la literatura se llama *spiral-down effect* y que en la
# planta se ve como un producto que «ya no se vende» cuando en realidad hace
# meses que no hay para vender.
#
# Aquí se mide el tamaño del sesgo. Nótese que hay que separar dos causas
# distintas de despacho bajo — el quiebre de stock y el cierre de planta — porque
# se tratan de manera diferente.

# %%
sin_cierre = dem[dem.planta_cerrada == 0]

sesgo_global = 1 - sin_cierre.despachos_cajas.sum() / sin_cierre.pedidos_cajas.sum()
sesgo_con_cierre = 1 - dem.despachos_cajas.sum() / dem.pedidos_cajas.sum()
n_cens = int(verdad.censurado.sum())

print(f"Filas censuradas (columna docente 'censurado'): {n_cens} de {len(verdad)} "
      f"= {n_cens/len(verdad):.1%}")
print(f"Subestimación del volumen usando despachos, sin el cierre : {sesgo_global:.2%}")
print(f"Subestimación si además se cuentan las semanas de cierre  : {sesgo_con_cierre:.2%}")

print("\nEl promedio global esconde la heterogeneidad. Por SKU:")
print(f"  {'sku':<10}{'sesgo':>9}{'semanas cens.':>15}")
for s in sorted(dem.sku.unique()):
    d = sin_cierre[sin_cierre.sku == s]
    b = 1 - d.despachos_cajas.sum() / d.pedidos_cajas.sum()
    c = int(verdad[verdad.sku == s].censurado.sum())
    print(f"  {s:<10}{b:>9.2%}{c:>15}")

# %% [markdown]
# Los 2,67 % globales parecen inofensivos y por eso son peligrosos: nadie va a
# rechazar un pronóstico por un 2,7 %. Pero el promedio mezcla SKU muy distintos.
# POR-3000 pierde **28 %** de su volumen: es el formato institucional, su
# capacidad logística está dimensionada para el pedido típico y los pedidos de
# casino la desbordan. Un pronóstico de POR-3000 construido sobre despachos no
# está un poco bajo; está estructuralmente equivocado.
#
# La corrección defendible en clase es la más simple que se puede auditar en una
# planilla: sobre las semanas marcadas como censuradas, se sube el observado en
# una desviación local. No es óptima —existen modelos tobit y de verosimilitud
# censurada que lo hacen mejor— pero se explica en una lámina y deja rastro. Lo
# que **no** es defendible es ignorar la censura, porque equivale a afirmar que
# nunca faltó producto.

# %%
def corregir_censura(df, col_obs="despachos_cajas", col_out="demanda_estimada",
                     z=1.0, ventana=13):
    """Sube el despacho observado en z desviaciones locales donde hubo censura.

    Devuelve una copia y una bitácora. Un dato modificado sin registro es un
    dato inventado: cada corrección tiene que ser rastreable hasta la línea que
    la hizo.
    """
    out, bitacora = df.copy(), []
    out[col_out] = out[col_obs].astype(float)
    for k, g in out.groupby("sku"):
        s = g[col_obs].astype(float)
        sigma = s.rolling(ventana, center=True, min_periods=3).std().fillna(s.std())
        m = g.censurado == 1
        out.loc[g.index[m], col_out] = (s[m] + z * sigma[m]).to_numpy()
        if m.any():
            bitacora.append(f"sku={k}: {int(m.sum())} semanas subidas en "
                            f"{z} sigma local")
    return out, bitacora


dem = dem.merge(verdad[["t", "sku", "censurado"]], on=["t", "sku"], how="left")
dem_c, bit = corregir_censura(dem)
for b in bit:
    print("  ->", b)

recuperado = (dem_c.demanda_estimada.sum() / dem_c.despachos_cajas.sum() - 1)
print(f"\nLa corrección recupera {recuperado:.2%} del volumen despachado.")
print("Sigue por debajo de los pedidos reales: la censura destruye información")
print("y ninguna corrección la devuelve entera. La corrige, no la deshace.")

# %% [markdown]
# ## 2. El error de unidad, y por qué un detector ingenuo lo arruina
#
# En t = 87 el SKU TOM-480 registra 40.680 cajas donde sus vecinas registran
# unas 2.000. La respuesta refleja es «es un dato atípico, lo recorto». La
# respuesta correcta empieza por una pregunta distinta: **¿40.680 es un número
# raro, o es el mismo número en otra unidad?**
#
# La diferencia no es filosófica. Un pico real de demanda —un contrato, una
# promoción grande— es información que hay que conservar. Un error de unidad es
# un registro cargado en unidades en vez de cajas, y se arregla **dividiendo**,
# no recortando: winsorizar reemplaza 40.680 por el percentil 99, que no es ni
# el valor real ni el valor registrado, sino un número inventado.
#
# El criterio que distingue los dos casos tiene dos partes, y las dos son
# necesarias:
#
# 1. el valor supera en más de **3 veces** la mediana local (hay algo que mirar);
# 2. existe un factor *f* **que existe en el maestro de productos** —unidades por
#    caja, cajas por pallet— tal que dividir por *f* devuelve el valor dentro de
#    3 MAD locales.
#
# La segunda condición es la que hace el trabajo. Un pico de demanda real no se
# arregla dividiendo por 24: queda igual de raro, solo que 24 veces más chico.

# %%
def clasificar_intermitencia(serie):
    """Clasificación de Syntetos-Boylan-Croston por ADI y CV^2.

    ADI  = periodos / periodos con demanda     (qué tan espaciada)
    CV^2 = (sd / media de los no nulos)^2      (qué tan variable el tamaño)
    """
    s = pd.Series(serie).dropna().astype(float)
    nz = s[s > 0]
    if len(nz) == 0:
        return {"ADI": np.nan, "CV2": np.nan, "p_ceros": 1.0, "clase": "sin demanda"}
    adi = len(s) / len(nz)
    cv2 = float((nz.std(ddof=1) / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    if adi < 1.32 and cv2 < 0.49:
        clase = "suave"
    elif adi < 1.32:
        clase = "errática"
    elif cv2 < 0.49:
        clase = "intermitente"
    else:
        clase = "grumosa"
    return {"ADI": float(adi), "CV2": cv2, "p_ceros": float((s == 0).mean()),
            "clase": clase}


def detectar_error_unidad(df, col="pedidos_cajas", factores=(6, 12, 24),
                          ventana=9, umbral_salto=3.0, k_mad=3.0,
                          omitir_intermitentes=True):
    """Busca observaciones que sean un MÚLTIPLO CONOCIDO de su vecindario."""
    filas = []
    for k, g in df.groupby("sku"):
        g = g.sort_values("t")
        if omitir_intermitentes:
            if clasificar_intermitencia(g[col])["clase"] in (
                    "intermitente", "grumosa", "sin demanda"):
                continue
        mediana = g[col].rolling(ventana, center=True, min_periods=3).median()
        mad = (g[col] - mediana).abs().rolling(
            ventana, center=True, min_periods=3).median()
        mad = mad.replace(0, np.nan).fillna(g[col].std() or 1.0)
        for i, r in g.iterrows():
            med, m_i = mediana.loc[i], mad.loc[i]
            if not np.isfinite(med) or med <= 0 or r[col] < umbral_salto * med:
                continue
            for f in sorted(factores):                 # el f más pequeño que sirva
                if abs(r[col] / f - med) <= k_mad * m_i:
                    filas.append({"sku": k, "t": int(r.t), "valor": float(r[col]),
                                  "mediana_local": float(med), "mad_local": float(m_i),
                                  "factor_detectado": int(f),
                                  "valor_corregido": float(r[col] / f)})
                    break
    return pd.DataFrame(filas)


det = detectar_error_unidad(dem)
print("Criterio completo (con la guarda de intermitencia):")
print(det.to_string(index=False))

# %% [markdown]
# La detección se confirma sola: el factor que el algoritmo encontró **no** es un
# parámetro ajustado hasta que funcionara, es una columna del maestro de
# productos. Que coincidan es la evidencia.

# %%
upc = dict(zip(maestro.sku, maestro.unidades_por_caja))
for _, r in det.iterrows():
    print(f"  {r.sku}: factor detectado = {int(r.factor_detectado)} | "
          f"unidades_por_caja en el maestro = {upc[r.sku]} | "
          f"{'coinciden' if int(r.factor_detectado) == upc[r.sku] else 'NO coinciden'}")
    print(f"     {r.valor:,.0f} / {int(r.factor_detectado)} = "
          f"{r.valor_corregido:,.0f} cajas, contra una mediana local de "
          f"{r.mediana_local:,.0f}")

# %% [markdown]
# ### La advertencia: la misma regla, aplicada donde no corresponde
#
# Ahora se apaga la guarda de intermitencia y se corre exactamente el mismo
# criterio sobre las seis series.

# %%
sin_guarda = detectar_error_unidad(dem, omitir_intermitentes=False)
print(f"Detecciones sin la guarda: {len(sin_guarda)}")
print(sin_guarda.groupby("sku").size().to_string())
extra = sin_guarda[sin_guarda.sku != "TOM-480"]
print(f"\n{len(extra)} falsos positivos, TODOS en POR-3000.")
print(extra.head(8).to_string(index=False))

vol_antes = dem[dem.sku == "POR-3000"].pedidos_cajas.sum()
vol_despues = vol_antes - (extra.valor - extra.valor_corregido).sum()
print(f"\nSi se 'corrigieran': el volumen anual de POR-3000 pasaría de "
      f"{vol_antes:,.0f} a {vol_despues:,.0f} cajas "
      f"({vol_despues/vol_antes-1:.1%}).")

# %% [markdown]
# Veintidós falsos positivos y un tercio del volumen del SKU borrado, con una
# regla que **es correcta**. El problema no es la regla: es la serie. POR-3000
# tiene 35 % de semanas en cero y los pedidos llegan en lotes, así que su mediana
# local vale casi nada y cualquier pedido institucional legítimo supera «3 veces
# la mediana» sin esfuerzo. La condición (2) tampoco filtra, porque con un MAD
# local minúsculo cualquier división cae dentro de 3 MAD.
#
# La lección se generaliza más allá de este caso: **una regla de limpieza es
# válida solo dentro del régimen de serie para el que fue diseñada**, y ese
# régimen hay que verificarlo antes, no después. La clasificación ADI/CV² de la
# sección 6 no es un adorno taxonómico: es la guarda que hace que esta regla no
# destruya datos.
#
# Ahora sí, se corrige el único error real y se deja registro.

# %%
lim = dem.copy()
bitacora = []
for _, r in det.iterrows():
    m = (lim.sku == r.sku) & (lim.t == r.t)
    for c in ("pedidos_cajas", "despachos_cajas"):
        antes = float(lim.loc[m, c].iloc[0])
        lim.loc[m, c] = antes / r.factor_detectado
        bitacora.append(f"{r.sku} t={int(r.t)} {c}: {antes:,.0f} / "
                        f"{int(r.factor_detectado)} = {antes/r.factor_detectado:,.0f}")
for b in bitacora:
    print("  ->", b)

# %% [markdown]
# ## 3. Ceros estructurales contra intermitencia real
#
# En la serie hay dos tipos de cero que se escriben igual en el CSV y significan
# cosas opuestas:
#
# * **Cero estructural.** Las semanas ISO 5 y 6 la planta cierra por mantención
#   mayor. El despacho es cero porque no había quién despachara, no porque nadie
#   pidiera. El pedido de esas semanas es positivo y además se arrastra: parte se
#   recupera la semana siguiente. Como dato de demanda, ese cero **no existe**:
#   es un faltante.
# * **Intermitencia real.** POR-3000 tiene semanas en cero porque efectivamente
#   no hubo pedido. Ese cero **sí** es demanda cero y es información: es
#   exactamente lo que Croston va a modelar.
#
# Distinguirlos no es un detalle de implementación, es la decisión. Meter 36
# ceros estructurales en la serie hunde el nivel de enero y contamina el índice
# estacional de esas semanas para siempre, porque el suavizamiento estacional las
# va a promediar con los años buenos. Y al revés: convertir en faltantes los
# ceros de POR-3000 borraría el fenómeno que se quiere pronosticar.
#
# El criterio que los separa es externo a la serie: existe una bandera
# `planta_cerrada` que viene del calendario de la planta. Cuando esa bandera no
# existe —el caso habitual— hay que ir a buscarla al área, no inferirla.

# %%
cierre = lim[lim.planta_cerrada == 1]
print("Semanas de cierre: ISO "
      f"{sorted(int(x) for x in cal[cal.planta_cerrada == 1].iso_semana.unique())}")
print(f"{len(cierre)} filas semana-SKU con la planta cerrada")
print(f"  despacho medio esas semanas : {cierre.despachos_cajas.mean():,.1f} cajas")
print(f"  pedido medio esas semanas   : {cierre.pedidos_cajas.mean():,.1f} cajas")
print("  el pedido NO es cero: la demanda existió, la capacidad de despacho no.")

ceros_por = lim[lim.sku == "POR-3000"]
ceros_estructurales = int((ceros_por.planta_cerrada == 1).sum())
ceros_totales = int((ceros_por.pedidos_cajas == 0).sum())
print(f"\nPOR-3000: {ceros_totales} semanas con PEDIDO cero de {len(ceros_por)} "
      f"({ceros_totales/len(ceros_por):.1%}).")
print(f"  de ellas, {ceros_estructurales} caen en semana de cierre y "
      f"{ceros_totales - ceros_estructurales} son intermitencia genuina.")

# %% [markdown]
# El tratamiento correcto es marcar como faltante (`NaN`) el cero que tiene causa
# conocida, y después imputarlo con el **mismo período del ciclo anterior**, no
# con una interpolación lineal. Interpolar sobre una serie estacional aplana el
# ciclo justo en el tramo imputado; si el hueco cae en un peak, se pierde el
# peak. Aquí el hueco cae en enero, que para conservas es valle, pero el
# argumento es el mismo.

# %%
def marcar_e_imputar(df, col="despachos_cajas", m=52):
    out, bit = df.copy(), []
    mask = (out.planta_cerrada == 1) & (out[col] <= 0)
    out.loc[mask, col] = np.nan
    bit.append(f"{int(mask.sum())} observaciones de '{col}' marcadas como faltantes")
    for k, g in out.groupby("sku"):
        g = g.sort_values("t")
        s = g[col]
        faltan = s.isna()
        if not faltan.any():
            continue
        rell = s.where(~faltan, s.shift(m))          # mismo período del año anterior
        rell = rell.where(~rell.isna(),              # si no hay ciclo previo
                          s.rolling(9, center=True, min_periods=1).median())
        out.loc[g.index, col] = rell.to_numpy()
        bit.append(f"  sku={k}: {int(faltan.sum())} faltantes imputados con y[t-{m}]")
    return out, bit


lim, bit = marcar_e_imputar(lim)
for b in bit[:4]:
    print("  ->", b)

# %% [markdown]
# ## 4. El quiebre estructural, y por qué el detector obvio lo encuentra mal
#
# En t ≈ 120 la familia TOM gana un contrato con una cadena de supermercados y su
# nivel sube un escalón. Un quiebre de nivel es un problema de pronóstico serio
# porque **invalida la historia anterior**: los datos de antes del quiebre
# describen una empresa que ya no existe, y un modelo estimado sobre toda la
# serie promedia dos regímenes y no representa ninguno.
#
# Detectarlo se hace con un test de Chow por fuerza bruta: para cada corte
# candidato se ajustan dos medias y se mide cuánto baja la suma de cuadrados
# residual; gana el corte que más la reduce. Es simple y, sobre la serie cruda,
# está **garantizadamente mal**.
#
# El motivo es que sobre una serie estacional el mayor salto de media no está
# donde cambia el nivel: está donde cambia la **estación**. El detector encuentra
# el punto del año donde la serie pasa del valle al peak, que es un salto mucho
# más grande, y lo reporta como quiebre estructural.
#
# La solución no es un test más sofisticado: es cambiar la serie sobre la que se
# corre. La razón interanual y_t / y_{t−52} es aproximadamente constante bajo
# estacionalidad —cada semana se compara consigo misma— y salta exactamente donde
# está el escalón de nivel.

# %%
def detectar_quiebre(serie, minimo_tramo=20, m=None):
    """Chow por fuerza bruta. Con m, se corre sobre la razón interanual."""
    y0 = pd.Series(serie).dropna().astype(float)
    if m:
        v = y0.to_numpy()
        razon = v[m:] / np.where(v[:-m] == 0, np.nan, v[:-m])
        y = pd.Series(razon).dropna().to_numpy()
        desfase = m + (len(v) - m - len(y))
    else:
        y, desfase = y0.to_numpy(), 0
    n = len(y)
    if n < 2 * minimo_tramo:
        return {"quiebre": None}
    sct = float(((y - y.mean()) ** 2).sum())
    mejor, mejor_t = sct, None
    for c in range(minimo_tramo, n - minimo_tramo):
        a, b = y[:c], y[c:]
        scr = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if scr < mejor:
            mejor, mejor_t = scr, c
    if mejor_t is None:
        return {"quiebre": None}
    a, b = y[:mejor_t], y[mejor_t:]
    return {"quiebre": int(mejor_t + desfase),
            "salto_relativo": float(b.mean() / a.mean() - 1.0),
            "reduccion_sc": float(1 - mejor / sct),
            "sobre": "razón interanual" if m else "serie cruda"}


tom_sucia = dem[dem.sku == "TOM-480"].sort_values("t").pedidos_cajas
tom_limpia = lim[lim.sku == "TOM-480"].sort_values("t").pedidos_cajas

print(f"{'serie':<22}{'transformación':<20}{'t':>6}{'salto':>10}{'R2 del corte':>14}")
for etiqueta, s in [("SIN limpiar", tom_sucia), ("limpia", tom_limpia)]:
    for m in (None, 52):
        r = detectar_quiebre(s, m=m)
        tr = "razón interanual" if m else "ninguna (cruda)"
        print(f"{etiqueta:<22}{tr:<20}{r['quiebre']:>6}"
              f"{r['salto_relativo']:>+10.1%}{r['reduccion_sc']:>14.3f}")

quiebre = detectar_quiebre(tom_limpia, m=52)

# %% [markdown]
# Las cuatro filas cuentan la historia completa, y conviene leerlas despacio.
#
# * **Sin limpiar, cualquiera de las dos transformaciones devuelve t = 86.** No
#   es casualidad: es el error de unidad de t = 87, que por sí solo domina toda
#   la suma de cuadrados. El detector encontró el outlier, no el quiebre. Y lo
#   hizo sin quejarse, devolviendo un número perfectamente creíble.
# * **Limpia pero sin razón interanual: t = 108, +48,5 %.** Doce semanas antes
#   del quiebre real y con un salto inflado en quince puntos. Ahí es donde la
#   estacionalidad de TOM pasa del valle al peak.
# * **Limpia y sobre la razón interanual: t = 121, +33,0 %.** El generador plantó
#   el quiebre en t = 120 con un salto de +35 %, compuesto con una tendencia
#   anual de +6 %. El resto de la diferencia la explican la censura y el cierre
#   de planta, que también tocan esas semanas.
#
# La moraleja operativa: un número que sale de un procedimiento correcto no es
# necesariamente correcto. Este detector es el mismo en los cuatro casos; lo que
# cambia es lo que se le da de comer.

# %% [markdown]
# ## 5. Feriados: uno fijo y uno móvil
#
# Fiestas Patrias cae siempre en la semana ISO 38 y arrastra un efecto de resaca
# en la 39: la demanda se adelanta, no se crea. Una variable indicadora de semana
# ISO la captura sin problemas.
#
# Semana Santa **no tiene semana ISO fija**. Depende del calendario litúrgico y
# en estos cuatro años cae en tres semanas distintas. El efecto en Itata es real
# —en cuaresma sube el consumo de conservas vegetales, 1,20× en TOM y 1,12× en
# POR— pero una indicadora de semana ISO fija lo reparte entre tres semanas y
# estima tres efectos pequeños en vez de uno grande.
#
# Este es el caso general de los feriados móviles: Semana Santa, Año Nuevo Chino,
# Ramadán, Acción de Gracias, el Cyber. Ninguno se captura con una dummy de
# semana del año. El regresor hay que construirlo desde el calendario, y ese
# calendario es un insumo externo que hay que conseguir y mantener.

# %%
ss = cal[cal.semana_santa == 1][["iso_anio", "iso_semana", "semana"]]
print("Semana Santa por año (semana ISO):")
print(ss.to_string(index=False))
print(f"\nSemanas ISO ocupadas: {sorted(int(x) for x in ss.iso_semana.unique())} "
      "— tres distintas en cuatro años.")

# efecto medido sobre la señal desestacionalizada, como lo haría un analista
m = lim.merge(verdad[["t", "sku", "nivel", "estacionalidad"]], on=["t", "sku"])
m["ajustado"] = m.pedidos_cajas / (m.nivel * m.estacionalidad)
d = m[(m.promo == 0) & (m.planta_cerrada == 0) & (m.sku != "POR-3000")]

print(f"\n{'familia':<10}{'Fiestas Patrias':>18}{'Semana Santa':>16}")
for fam in ["DUR", "TOM", "POR"]:
    g = d[d.familia == fam]
    base = g[(g.fiestas_patrias == 0) & (g.semana_santa == 0)].ajustado.mean()
    print(f"{fam:<10}{g[g.fiestas_patrias == 1].ajustado.mean()/base:>17.2f}x"
          f"{g[g.semana_santa == 1].ajustado.mean()/base:>15.2f}x")

# contraste: qué pasa con una dummy de semana ISO fija
iso_moda = int(ss.iso_semana.mode().iloc[0])
g = d[d.familia == "TOM"]
base = g[(g.fiestas_patrias == 0) & (g.semana_santa == 0)].ajustado.mean()
fija = g[g.iso_semana == iso_moda].ajustado.mean() / base
movil = g[g.semana_santa == 1].ajustado.mean() / base
print(f"\nTOM, efecto de Semana Santa estimado con...")
print(f"  el regresor móvil (calendario litúrgico) : {movil:.3f}x")
print(f"  una dummy fija en la semana ISO {iso_moda}        : {fija:.3f}x")
print(f"  se pierde {1 - (fija-1)/(movil-1):.0%} del efecto por usar la dummy fija.")

# %% [markdown]
# ## 6. Clasificación ADI/CV²: el cuadrante decide el método
#
# Antes de elegir un modelo hay que saber con qué tipo de serie se está tratando.
# La clasificación de Syntetos, Boylan y Croston usa dos números:
#
# * **ADI** (*average demand interval*) = períodos / períodos con demanda. Mide
#   qué tan **espaciada** es la demanda. Vale 1 si todas las semanas tienen
#   pedido.
# * **CV²** = (desviación / media de los valores **no nulos**)². Mide qué tan
#   variable es el **tamaño** del pedido cuando hay pedido.
#
# Los cortes clásicos son ADI = 1,32 y CV² = 0,49, y dejan cuatro regiones:
#
# | región | ADI | CV² | qué usar |
# |---|---|---|---|
# | **suave** | < 1,32 | < 0,49 | suavizamiento exponencial, ETS, Holt-Winters |
# | **errática** | < 1,32 | ≥ 0,49 | ETS robusto, atención a los outliers |
# | **intermitente** | ≥ 1,32 | < 0,49 | Croston |
# | **grumosa** | ≥ 1,32 | ≥ 0,49 | SBA / TSB; y aceptar que el error será alto |
#
# Lo que hay que entender es la dirección de la flecha: **el cuadrante decide el
# método, no al revés**. Aplicar Holt-Winters a una serie grumosa no es una
# elección discutible; es un error de tipo, y el cuaderno 02 lo va a mostrar con
# números. La clasificación es además la guarda que salvó a POR-3000 de la
# limpieza de la sección 2.

# %%
clases = []
for s in sorted(lim.sku.unique()):
    c = clasificar_intermitencia(lim[lim.sku == s].sort_values("t").pedidos_cajas)
    c["sku"] = s
    c["canal"] = lim[lim.sku == s].canal.iloc[0]
    clases.append(c)
clases = pd.DataFrame(clases)[["sku", "canal", "ADI", "CV2", "p_ceros", "clase"]]
print(clases.round(3).to_string(index=False))

print("\nContraste — TOM-480 clasificado ANTES de corregir el error de unidad:")
c_sucia = clasificar_intermitencia(tom_sucia)
print(f"  ADI = {c_sucia['ADI']:.2f}  CV2 = {c_sucia['CV2']:.3f}  "
      f"-> clase '{c_sucia['clase']}'")
print("  Un solo registro mal cargado cambia el diagnóstico de la serie y, con")
print("  él, la familia de métodos que se iba a considerar.")

# %% [markdown]
# ## 7. Los dos gráficos que resumen el diagnóstico

# %%
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

# --- (a) las cuatro regiones de Syntetos-Boylan-Croston
a = ax[0, 0]
ADI_C, CV2_C = 1.32, 0.49
xmax, ymax = 1.85, 1.15
a.axhspan(0, CV2_C, xmin=0, xmax=(ADI_C - 0.9) / (xmax - 0.9), color="#2e7d32", alpha=0.10)
a.axhspan(CV2_C, ymax, xmin=0, xmax=(ADI_C - 0.9) / (xmax - 0.9), color="#f9a825", alpha=0.13)
a.axhspan(0, CV2_C, xmin=(ADI_C - 0.9) / (xmax - 0.9), xmax=1, color="#1565c0", alpha=0.10)
a.axhspan(CV2_C, ymax, xmin=(ADI_C - 0.9) / (xmax - 0.9), xmax=1, color="#c62828", alpha=0.12)
a.axvline(ADI_C, color="#37474f", lw=1.2, ls="--")
a.axhline(CV2_C, color="#37474f", lw=1.2, ls="--")
for txt, (px, py), col in [("suave\n(ETS, Holt-Winters)", (0.97, 0.10), "#1b5e20"),
                           ("errática\n(ETS robusto)", (0.97, 0.95), "#e65100"),
                           ("intermitente\n(Croston)", (1.38, 0.10), "#0d47a1"),
                           ("grumosa\n(SBA / TSB)", (1.38, 0.95), "#b71c1c")]:
    a.text(px, py, txt, color=col, fontsize=8.5, va="top", fontweight="bold")
for _, r in clases.iterrows():
    a.plot(r.ADI, r.CV2, "o", ms=10, color="#37474f", zorder=4)
    a.annotate(r.sku, (r.ADI, r.CV2), textcoords="offset points", xytext=(9, -3),
               fontsize=9)
a.plot(c_sucia["ADI"], c_sucia["CV2"], "X", ms=12, color="#c62828", zorder=5)
a.annotate("TOM-480\nsin limpiar", (c_sucia["ADI"], c_sucia["CV2"]),
           textcoords="offset points", xytext=(9, -14), fontsize=8, color="#c62828")
a.set_xlim(0.9, xmax)
a.set_ylim(0, ymax)
a.set_xlabel("ADI  (períodos / períodos con demanda)")
a.set_ylabel("CV²  de los valores no nulos")
a.set_title("(a) Clasificación Syntetos-Boylan-Croston", loc="left")

# --- (b) censura: pedidos contra despachos en el SKU más afectado
b = ax[0, 1]
d = dem[dem.sku == "POR-3000"].sort_values("t")
b.plot(d.t, d.pedidos_cajas, lw=1.1, color="#37474f", label="pedidos (demanda real)")
b.plot(d.t, d.despachos_cajas, lw=1.1, color="#c62828", label="despachos (observado)")
cen = d[d.censurado == 1]
b.plot(cen.t, cen.despachos_cajas, "o", ms=3.5, color="#c62828", mfc="none",
       label=f"censurado (n = {len(cen)})")
b.fill_between(d.t, d.despachos_cajas, d.pedidos_cajas, color="#c62828", alpha=0.10)
b.set_title("(b) POR-3000: la censura se come el 28 % del volumen", loc="left")
b.set_xlabel("semana t")
b.set_ylabel("cajas")
b.legend(fontsize=8, ncol=1)

# --- (c) el error de unidad
c_ax = ax[1, 0]
d = dem[dem.sku == "TOM-480"].sort_values("t")
c_ax.plot(d.t, d.pedidos_cajas, lw=1.1, color="#37474f")
c_ax.plot([87], [d[d.t == 87].pedidos_cajas.iloc[0]], "o", ms=12, mfc="none",
          mew=2.2, color="#c62828")
c_ax.annotate("t = 87: 40.680 cajas\n= 1.695 × 24 unidades/caja",
              (87, d[d.t == 87].pedidos_cajas.iloc[0]),
              textcoords="offset points", xytext=(14, -26), fontsize=9,
              color="#c62828")
c_ax.set_yscale("log")
c_ax.set_title("(c) TOM-480 en escala log: un factor 24, no un pico", loc="left")
c_ax.set_xlabel("semana t")
c_ax.set_ylabel("cajas (log)")

# --- (d) el quiebre estructural visto en la razón interanual
d_ax = ax[1, 1]
v = tom_limpia.to_numpy()
raz = v[52:] / np.where(v[:-52] == 0, np.nan, v[:-52])
tt = np.arange(53, len(v) + 1)
d_ax.plot(tt, raz, lw=1.0, color="#37474f", label="y[t] / y[t-52]")
q = quiebre["quiebre"]
d_ax.axhline(np.nanmean(raz[tt < q]), xmin=0, xmax=(q - 53) / (len(v) - 53),
             color="#1565c0", lw=2.0, label="media antes / después")
d_ax.axhline(np.nanmean(raz[tt >= q]), xmin=(q - 53) / (len(v) - 53), xmax=1,
             color="#1565c0", lw=2.0)
d_ax.axvline(q, color="#c62828", lw=1.6, ls="--")
d_ax.axvline(120, color="#2e7d32", lw=1.2, ls=":")
d_ax.text(q + 2, np.nanmax(raz) * 0.95,
          f"detectado t = {q}\nsalto {quiebre['salto_relativo']:+.1%}",
          fontsize=9, color="#c62828")
d_ax.text(120 - 34, np.nanmin(raz) * 1.05, "real t = 120 (+35 %)", fontsize=8,
          color="#2e7d32")
d_ax.axhline(1.0, color="#9e9e9e", lw=0.8)
d_ax.set_title("(d) TOM-480: el quiebre aparece en la razón interanual", loc="left")
d_ax.set_xlabel("semana t")
d_ax.set_ylabel("razón interanual")
d_ax.legend(fontsize=8)

figura(fig, "diagnostico_itata")
plt.show()

# %% [markdown]
# El panel (c) está en escala logarítmica a propósito: en escala lineal el único
# punto visible es el de t = 87 y la serie se aplasta contra el eje. En log se ve
# que el punto está exactamente un factor multiplicativo por encima del resto,
# que es la firma de un error de unidad y no de un pico de demanda.
#
# El panel (d) es el que hay que llevarse. La serie de la razón interanual es
# plana alrededor de 1 durante los primeros dos años y salta a ≈ 1,33 en t = 121.
# Sobre la serie original ese escalón es invisible: está tapado por un ciclo
# anual de ±22 % de amplitud.

# %% [markdown]
# ## 8. Comprobación contra las cifras de referencia

# %%
print("Comprobación:")
ok = [
    verificar(len(det), 1, "errores de unidad con la guarda"),
    verificar(len(sin_guarda), 23, "detecciones sin la guarda"),
    verificar(len(extra), 22, "falsos positivos en el SKU grumoso"),
    verificar(int(det.factor_detectado.iloc[0]), 24, "factor detectado (= unidades/caja)"),
    verificar(quiebre["quiebre"], 121, "t del quiebre (razón interanual)"),
    verificar(quiebre["salto_relativo"], 0.330, "salto del quiebre", tol=2e-2),
    verificar(detectar_quiebre(tom_limpia)["quiebre"], 108, "t del quiebre (serie cruda)"),
    verificar(n_cens, 107, "filas censuradas"),
    verificar(sesgo_global, 0.0267, "sesgo por usar despachos", tol=5e-3),
    verificar(float(clases.loc[clases.sku == "POR-3000", "ADI"].iloc[0]), 1.54,
              "ADI de POR-3000", tol=5e-3),
    verificar(float(clases.loc[clases.sku == "POR-3000", "CV2"].iloc[0]), 0.620,
              "CV2 de POR-3000", tol=5e-3),
]
print(f"\n{sum(ok)}/{len(ok)} comprobaciones correctas")

assert list(clases[clases.clase == "grumosa"].sku) == ["POR-3000"], \
    "POR-3000 debe ser la única serie grumosa"
assert (clases[clases.sku != "POR-3000"].clase == "suave").all(), \
    "las otras cinco series deben quedar suaves tras la limpieza"

tabla(lim, "itata_limpia")
tabla(clases, "clasificacion_sbc")
resumen({"errores_unidad": len(det), "falsos_positivos_sin_guarda": len(extra),
         "quiebre_t": quiebre["quiebre"], "quiebre_salto": quiebre["salto_relativo"],
         "filas_censuradas": n_cens, "sesgo_despachos": sesgo_global,
         "clases": dict(zip(clases.sku, clases.clase))}, "resumen_limpieza")
print("\nGuardado en resultados/ — la serie limpia alimenta los cuadernos 02 a 05.")

# %% [markdown]
# ---
#
# ## Para llevarse
#
# 1. **Pronosticar despachos sesga a la baja y el sesgo se acumula.** La censura
#    solo actúa en una dirección, así que no se promedia con el tiempo: se suma.
#    Y se retroalimenta, porque el plan bajo produce más quiebres. En POR-3000 el
#    sesgo es del 28 %, no del 2,7 % del promedio.
# 2. **Un valor 24 veces más alto no es un outlier: es otra unidad.** La prueba
#    no es que el valor sea raro, es que el factor que lo normaliza esté en el
#    maestro de productos. Se corrige dividiendo; recortar inventa un dato.
# 3. **Una regla correcta aplicada fuera de su régimen destruye datos.** El mismo
#    detector, sin la guarda de intermitencia, marca 22 pedidos legítimos de
#    POR-3000 y borra un tercio de su volumen. Clasificar la serie va antes de
#    limpiarla.
# 4. **Primero se limpia, después se diagnostica.** Sobre la serie sucia el
#    detector de quiebre encuentra el outlier (t = 86); sobre la serie limpia sin
#    transformar encuentra el cambio de estación (t = 108); solo sobre la razón
#    interanual de la serie limpia encuentra el quiebre (t = 121, +33 %).
#
# ## Ejercicios
#
# 1. Cambie `z` de 1,0 a 0,5 y a 2,0 en `corregir_censura` y recalcule el volumen
#    recuperado por SKU. ¿Con qué valor de `z` la serie corregida de POR-3000
#    recupera el volumen de `pedidos_cajas`? Discuta por qué elegir `z` mirando la
#    respuesta correcta —que en la planta no se conoce— no es un método.
# 2. Corra `detectar_quiebre` con `m = 52` sobre las otras cinco series limpias.
#    ¿En cuáles encuentra un quiebre con reducción de suma de cuadrados
#    comparable a la de TOM-480? Para las que no, ¿qué devuelve el detector, y por
#    qué reportar ese número como «quiebre» sería un error?
# 3. Construya la serie de despachos **sin** marcar los ceros de cierre de planta
#    (es decir, dejándolos como demanda cero), calcule el índice estacional de las
#    semanas ISO 5 y 6 con un promedio móvil centrado de 52 semanas, y compárelo
#    con el de la serie tratada. ¿Cuántas cajas de diferencia implica eso en el
#    plan de producción de enero para los seis SKU juntos?
