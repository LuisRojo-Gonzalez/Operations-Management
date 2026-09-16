# Scripts del curso — Gestión de Producción de Bienes y Servicios

Cuadernos Jupyter con **los modelos del curso resueltos sobre los datos de
clase** (Conservas del Itata S.A.). Están pensados como guía de estudio: cada
uno corre de principio a fin, muestra sus resultados y los contrasta contra la
cifra de referencia del curso.

No reemplazan a `Ejercicios/` (las cuatro guías de 20 ejercicios de lápiz y
papel) ni a `Laboratorio/` (los ocho laboratorios con sus propios datos). Son la
tercera vía: **ver el modelo funcionando**.

---

## Cómo se usa

```bash
pip install -r requirements.txt
jupyter lab            # o abrir la carpeta en VS Code
```

Los cuadernos se abren **desde su propia carpeta** (`Scripts/3_Planificacion/`,
por ejemplo). No hace falta configurar rutas: `comun.py` encuentra
`DatosClases/` subiendo por el árbol de carpetas, así que el repositorio se
puede mover o clonar en cualquier lugar.

En Colab, suba la carpeta de la unidad junto con una subcarpeta `datos/` que
tenga los CSV de esa unidad; `comun.py` la prefiere si existe.

Cada cuaderno escribe sus tablas, figuras y cifras en `resultados/`, al lado del
cuaderno.

---

## Contenido

### `1_Calidad/` — scikit-learn donde aporta

| cuaderno | qué muestra |
|---|---|
| `01_cartas_control` | X̄–R, Fase I contra Fase II, reglas de Nelson. Contraste con `IsolationForest`: el detector **se pierde el inicio del corrimiento** y marca tres subgrupos estables |
| `02_capacidad_proceso` | Cp, Cpk, Pp, Ppk; corto contra largo plazo; los 5,79 g por envase que la planta regala por miedo al mínimo legal |
| `03_msa_gage_rr` | R&R por rangos (AIAG); el instrumento se come 0,084 de Cp |
| `04_muestreo_aceptacion` | Carta p con n variable, Pareto, curva OC, AQL/LTPD/AOQL |

### `2_Pronostico/` — scikit-learn

| cuaderno | qué muestra |
|---|---|
| `01_limpieza_diagnostico` | Censura, error de unidad, quiebre estructural, ceros de cierre, ADI/CV² |
| `02_modelos_clasicos` | Ingenuos, SES, Holt, Holt-Winters, Croston/SBA escritos a mano |
| `03_regresion_sklearn` | Features de calendario y Fourier, `TimeSeriesSplit`, Ridge/RF/HistGB. Por qué la validación cruzada **aleatoria** es inválida aquí |
| `04_validacion_origen_rodante` | 12 orígenes, h = 1..4, WAPE/MASE/sesgo, y la σ̂(i,h) que alimenta la Unidad 3 |
| `05_jerarquia_reconciliacion` | Matriz S, bottom-up, top-down, MinT (OLS/WLS/shrink) |

### `3_Planificacion/` — Gurobi

| cuaderno | qué muestra |
|---|---|
| `01_uls_wagner_whitin` | PD exacta contra dos MIP. Gap de raíz **48,6 % / 99,8 % / 0 %** para el mismo óptimo |
| `02_clsp_capacidad` | CLSP multiproducto, cuello de botella, la semana focal que no cabe |
| `03_clsp_extensiones` | Horas extra, backlog, stock de seguridad, lead time, arranque. **El big-M ajustado deja de ser válido con backlog** |
| `04_politicas_carbono` | Cuatro políticas con el **mismo** presupuesto: el tope semanal cuesta 1,82× el global sin ahorrar un kilo |
| `05_glsp_cambios_secuencia` | Cambios dependientes de la secuencia. La contigüidad **no** es una desigualdad válida |
| `06_mlclsp_multinivel` | BOM de dos niveles; imputar las horas a ambos vuelve el modelo infactible |
| `07_estocastico_bietapa` | No anticipatividad, RP/EEV/WS, VSS y EVPI |
| `08_remanufactura_circular` | Tres inventarios; el caso de negocio **se da vuelta** al dividir por el rendimiento |
| `09_estocastico_multietapa` | Árbol de escenarios; el valor de la adaptación y por qué el contrafactual correcto no es la demanda promedio |

### `4_Secuenciamiento/` — Gurobi

| cuaderno | qué muestra |
|---|---|
| `01_una_maquina_reglas` | Cada regla alcanza el óptimo de **su** criterio y solo de ese |
| `02_setups_secuencia_atsp` | 1\|s_ij\|Cmax: big-M se queda en ~60 % de gap, ATSP+MTZ cierra en 0,02 s |
| `03_maquinas_paralelas` | Pm y Rm; secuenciar es en realidad **asignar** |
| `04_flowshop` | F3 con y sin permutación; Johnson y CDS |
| `05_jobshop` | Formulación disyuntiva; la cota se alcanza |
| `06_rcpsp_precedencias` | Precedencias generalizadas FS/SS/FF/SF con desfases de cualquier signo |
| `07_vrptw` | Ventanas de tiempo; con un técnico es infactible, y se demuestra **cuál par** se contradice |

---

## Cómo leer un cuaderno

Todos tienen la misma forma:

1. **Qué se decide aquí** — el problema en términos de planta, no de notación.
2. **Cifras de referencia** — una tabla con lo que el cuaderno debe reproducir.
3. **El desarrollo**, con el *porqué* antes de cada bloque de código.
4. **Comprobación** — `verificar(...)` contrasta cada resultado contra la
   referencia e imprime `OK` o `DIFIERE`. Si su corrida difiere, no está
   necesariamente mal, pero es una diferencia que hay que poder explicar.
5. **Para llevarse** y **Ejercicios** — tres preguntas que exigen modificar el
   cuaderno y volver a correrlo.

---

## Herramientas

`comun.py` — resolución de rutas, guardado de tablas y figuras, `verificar()`.
Los cuadernos son **autocontenidos**: escriben el modelo explícitamente con
`gurobipy` o `scikit-learn` en vez de llamar a una biblioteca del curso, porque
lo que hay que ver es la formulación.

`_fuentes/` y `_herramientas/construir.py` — los cuadernos se escriben como `.py`
anotados y se compilan con:

```bash
python _herramientas/construir.py                 # todas las unidades
python _herramientas/construir.py 3_Planificacion # una unidad
python _herramientas/construir.py --sin-ejecutar  # compilar sin correr
python _herramientas/construir.py --forzar        # regenerar aunque ya esté ejecutado
```

Un cuaderno ya ejecutado **no se regenera** salvo que su fuente sea más nuevo o
se pase `--forzar`: regenerarlo lo dejaría sin salidas, y en una guía de estudio
las salidas son el contenido.

---

## Licencia de Gurobi

`pip install gurobipy` trae una licencia limitada a **2000 variables y 2000
restricciones**. Todos los modelos de estos cuadernos caben, con una excepción
documentada en el propio cuaderno: el estocástico bietapa se corre con **6
escenarios** en vez de 8 (con 8 son 2184 variables). La licencia académica
—gratuita para universidades— levanta el límite.
