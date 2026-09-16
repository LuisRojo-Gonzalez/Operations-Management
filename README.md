<div align="center">

# 🏭 Gestión de Operaciones

**Calidad, pronóstico, planificación y secuenciamiento para la toma de decisiones en Ingeniería Industrial**

Material docente del curso *Gestión de Producción de Bienes y Servicios* — Departamento de Ingeniería Industrial, Universidad de Santiago de Chile (USACH).

[![License: CC0 1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/An%C3%A1lisis-Python-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
![Material](https://img.shields.io/badge/Material-PDF%20%2B%20CSV-blue.svg)
[![Universidad](https://img.shields.io/badge/USACH-Ingenier%C3%ADa%20Industrial-004b8d.svg)](https://www.usach.cl/)
![Estado](https://img.shields.io/badge/Estado-Activo-brightgreen.svg)

</div>

---

## 📌 Descripción

Este repositorio reúne el material del curso **Gestión de Producción de Bienes y Servicios** para estudiantes de Ingeniería Industrial. El curso estudia la operación como un sistema de transformación y conecta las decisiones de calidad, demanda, producción, inventario y programación de recursos.

> **Propósito del curso.** Analizar y mejorar sistemas de producción de bienes y servicios mediante herramientas de gestión, análisis de datos y optimización, fundamentando decisiones que consideren calidad, capacidad, costos, incertidumbre y nivel de servicio.

El material incluye **seis presentaciones**, **cuatro guías con 20 ejercicios cada una**, **solucionarios**, **datos para clases** y un **caso integrador con tres entregas**. El trabajo computacional utiliza **Python** y combina análisis estadístico, pronóstico y modelamiento de problemas de optimización.

Los ejemplos de clase se desarrollan en torno a **Conservas del Itata**, mientras que el proyecto integrador aborda la operación de **Bebidas del Maipo S.A.** Cada conjunto de datos corresponde a su propio contexto de trabajo.

---

## 🗂️ Temario

| # | Unidad | Contenidos principales | Diapositivas |
|---|---|---|---|
| 1 | **Conceptos básicos de la Gestión de Operaciones** | Sistemas de transformación, bienes y servicios, estrategia, procesos, evolución productiva y manufactura distribuida | [Conceptos](Slides/1_Conceptos.pdf) |
| 2 | **Calidad** | Mejora de procesos, herramientas de calidad, análisis del sistema de medición, control estadístico, capacidad y muestreo de aceptación | [Parte I](Slides/2_Calidad_I.pdf) · [Parte II](Slides/2_Calidad_II.pdf) |
| 3 | **Pronóstico y Planificación de Operaciones** | Diagnóstico de series, métodos ingenuos, suavizamiento exponencial, ETS, SARIMA, evaluación temporal, incertidumbre y jerarquías de pronóstico | [Pronóstico](Slides/3_Pronostico.pdf) |
| 4 | **Producción e Inventario** | Planificación agregada, lot-sizing, capacidad, MRP, modelos mixtos enteros, escenarios de incertidumbre y remanufactura | [Producción e inventario](Slides/4_Produccion_Inventario.pdf) |
| 5 | **Secuenciamiento** | Una máquina, máquinas paralelas, flow shop, job shop, precedencias, preparaciones, reglas de despacho, proyectos y servicios | [Secuenciamiento](Slides/5_Secuenciamiento.pdf) |

Las presentaciones completas están disponibles en [`Slides/`](Slides/). Las cuatro guías de ejercicios corresponden a las unidades 2 a 5.

---

## 📁 Estructura del repositorio

| Ruta | Contenido |
|---|---|
| [`Slides/`](Slides/) | Seis presentaciones del curso; Calidad se divide en dos partes |
| [`Ejercicios/Enunciados/`](Ejercicios/Enunciados/) | Guías de Calidad, Pronóstico, Planificación y Secuenciamiento |
| [`Ejercicios/Soluciones/`](Ejercicios/Soluciones/) | Solucionarios de las cuatro guías |
| [`DatosClases/1_Calidad/`](DatosClases/1_Calidad/) | Datos de calidad y scripts para generar datos y figuras |
| [`DatosClases/2_Pronostico/`](DatosClases/2_Pronostico/) | Series, calendario, maestro de productos, jerarquía y componentes de referencia |
| [`DatosClases/3_Planificacion/`](DatosClases/3_Planificacion/) | Datos de planificación de producción y remanufactura |
| [`DatosClases/4_Secuenciamiento/`](DatosClases/4_Secuenciamiento/) | Instancias de trabajos, máquinas, proyectos y visitas de técnicos |
| [`CaseStudy/Enunciado.pdf`](CaseStudy/Enunciado.pdf) | Encargo, organización, datos, requisitos y evaluación del caso integrador |
| [`CaseStudy/Datos/`](CaseStudy/Datos/) | Datos y metadatos de Bebidas del Maipo |
| [`CaseStudy/Rubricas/`](CaseStudy/Rubricas/) | Rúbricas de cátedra y laboratorio de las tres entregas |
| [`LICENSE`](LICENSE) | Texto de CC0 1.0 Universal |

---

## 🧪 Conjuntos de datos

### Datos para clases: Conservas del Itata

| Área | Archivos y aplicaciones |
|---|---|
| **Calidad** | `itata_peso.csv`, `itata_atributos.csv`, `itata_msa.csv` y `oc_plan.csv`: cartas de control, capacidad, medición y muestreo de aceptación |
| **Pronóstico** | `itata_demanda.csv`, `itata_calendario.csv`, `itata_maestro_sku.csv`, `itata_jerarquia.csv` e `itata_componentes_verdad.csv`: series, factores de calendario y estructura de productos |
| **Planificación** | Demanda, capacidad, inventarios, parámetros, recursos, tasas, cambios, insumos, lista de materiales y escenarios |
| **Remanufactura** | Horizonte, parámetros, procesos, escenarios y árbol de decisiones |
| **Secuenciamiento** | Instancias de una máquina, máquinas paralelas, flow shop, job shop, preparaciones dependientes de la secuencia, proyectos con recursos limitados y visitas de técnicos |

Los datos de planificación se encuentran en [`DatosClases/3_Planificacion/Planificiacion/`](DatosClases/3_Planificacion/Planificiacion/), con esa escritura en el nombre de la carpeta. Los de remanufactura están en [`DatosClases/3_Planificacion/Remanufactura/`](DatosClases/3_Planificacion/Remanufactura/).

### Datos del caso: Bebidas del Maipo

El caso trabaja con **9 SKU**, **3 familias de productos**, **156 semanas de historia** y un **horizonte de planificación de 13 semanas**. La programación detallada se centra en la semana 12 del horizonte, según [`metadatos.json`](CaseStudy/Datos/metadatos.json).

| Archivo | Contenido |
|---|---|
| [`maestro_sku.csv`](CaseStudy/Datos/maestro_sku.csv) | Productos, formatos, líneas, tasas de producción y parámetros económicos |
| [`demanda_semanal.csv`](CaseStudy/Datos/demanda_semanal.csv) | Ventas semanales, días sin stock, promociones y precios |
| [`calidad_llenado.csv`](CaseStudy/Datos/calidad_llenado.csv) | Mediciones de volumen por subgrupo, turno, línea y producto |
| [`calidad_atributos.csv`](CaseStudy/Datos/calidad_atributos.csv) | Inspecciones, unidades defectuosas y causas principales |
| [`especificaciones_calidad.csv`](CaseStudy/Datos/especificaciones_calidad.csv) | Contenidos nominales, límites y tolerancias utilizados en el caso |
| [`capacidad_horizonte.csv`](CaseStudy/Datos/capacidad_horizonte.csv) | Disponibilidad, tasas, horas extra, mantenciones y costos por recurso |
| [`inventario_inicial.csv`](CaseStudy/Datos/inventario_inicial.csv) | Existencias, niveles de servicio, tiempos de abastecimiento y lotes mínimos |
| [`setups_cip.csv`](CaseStudy/Datos/setups_cip.csv) | Tiempos y costos de limpieza entre familias |
| [`compromisos_semana_focal.csv`](CaseStudy/Datos/compromisos_semana_focal.csv) | Pedidos, fechas comprometidas, prioridades y multas por atraso |

El [enunciado](CaseStudy/Enunciado.pdf) describe los campos y su interpretación. Los CSV del caso utilizan **codificación UTF-8 y separador coma**.

> **Ventas y demanda.** Aunque el archivo se denomina `demanda_semanal.csv`, la variable observada es `ventas_cajas`. El análisis debe considerar `dias_sin_stock` antes de interpretar esas ventas como demanda.

### Reproducibilidad de los datos de Calidad

La carpeta de Calidad incluye [`generar_datos_clase.py`](DatosClases/1_Calidad/generar_datos_clase.py), que genera las cuatro bases de esa unidad con una **semilla predeterminada de `424242`** y muestra cálculos de referencia.

Desde la raíz del repositorio:

```bash
python3 -m pip install numpy pandas scipy
python3 DatosClases/1_Calidad/generar_datos_clase.py --seed 424242 --out DatosClases/1_Calidad
```

El script [`emitir_figuras.py`](DatosClases/1_Calidad/emitir_figuras.py) utiliza esos datos para producir macros de figuras en LaTeX/PGFPlots.

> ⚠️ El generador **reescribe los cuatro CSV de Calidad en la carpeta de salida**. Para conservar los archivos originales, indique otra carpeta mediante `--out`.

---

## 🚀 Guía de uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/LuisRojo-Gonzalez/Operations-Management.git
cd Operations-Management
```

También puede descargar el contenido desde **Code → Download ZIP** en GitHub.

### 2. Requisitos

- **Visor de PDF** para consultar diapositivas, guías, soluciones y rúbricas.
- **Calculadora u hoja de cálculo** para apoyar la resolución de ejercicios.
- **Python 3.10 o superior**, según los requisitos del caso integrador.
- **NumPy, pandas y SciPy** para ejecutar los scripts de Calidad.
- **StatsForecast** para reproducir los ejemplos de pronóstico que utilizan esa biblioteca.
- **Una herramienta de modelamiento y un solver** para los modelos de optimización. El enunciado contempla PuLP o Pyomo con CBC, HiGHS o Gurobi; cada equipo debe documentar su elección y sus dependencias.

Las bases están disponibles en el repositorio y pueden utilizarse directamente.

### 3. Empezar un ejercicio

Revise las diapositivas de la unidad y abra su guía en [`Ejercicios/Enunciados/`](Ejercicios/Enunciados/). Para una actividad computacional de Calidad, por ejemplo:

```python
# Ejecutar desde la raíz del repositorio
import pandas as pd

peso = pd.read_csv("DatosClases/1_Calidad/itata_peso.csv")
print(peso.head())
peso.info()
print(peso.describe(include="all"))
```

Para comenzar el caso integrador:

```python
import pandas as pd

ventas = pd.read_csv(
    "CaseStudy/Datos/demanda_semanal.csv",
    parse_dates=["fecha_lunes"],
)
print(ventas.head())
print(ventas.isna().sum())
```

En el proyecto, conserve los datos crudos y realice las transformaciones mediante código. El enunciado exige dependencias con versiones fijadas, semillas explícitas y un punto de entrada único documentado para ejecutar el análisis completo.

---

## 🎓 Guías de ejercicios

Las cuatro guías reúnen **80 ejercicios** sobre situaciones de Ingeniería Industrial. Cada una contiene **6 ejercicios básicos, 8 intermedios y 6 avanzados**.

| Guía | Enunciado | Solucionario |
|---|---|---|
| **Calidad** | [20 ejercicios](Ejercicios/Enunciados/1_Calidad.pdf) | [Soluciones](Ejercicios/Soluciones/1_Calidad.pdf) |
| **Pronóstico** | [20 ejercicios](Ejercicios/Enunciados/2_Pronostico.pdf) | [Soluciones](Ejercicios/Soluciones/2_Pronostico.pdf) |
| **Planificación** | [20 ejercicios](Ejercicios/Enunciados/3_Planificacion.pdf) | [Soluciones](Ejercicios/Soluciones/3_Planificacion.pdf) |
| **Secuenciamiento** | [20 ejercicios](Ejercicios/Enunciados/4_Secuenciamiento.pdf) | [Soluciones](Ejercicios/Soluciones/4_Secuenciamiento.pdf) |

Los ejercicios combinan reflexión conceptual, cálculos e interpretación operacional. Se solicita justificar decisiones, explicitar supuestos y explicar el significado de los resultados; **no se requieren demostraciones**.

---

## 🏭 Proyecto integrador: Bebidas del Maipo S.A.

El [caso integrador](CaseStudy/Enunciado.pdf) propone un proyecto acumulativo en **equipos de cuatro integrantes**. Las conclusiones de calidad y pronóstico alimentan la planificación; el plan de producción determina los lotes que luego deben secuenciarse.

| Entrega | Contenido | Ponderación dentro del caso | Rúbrica |
|---|---|---|---|
| **Entrega 1** | Calidad y pronóstico: diagnóstico, capacidad, demanda futura e incertidumbre | 25 % | [Entrega 1](CaseStudy/Rubricas/Entrega_1.pdf) |
| **Entrega 2** | Planificación de producción e inventario: formulación, capacidad, costos y análisis de resultados | 35 % | [Entrega 2](CaseStudy/Rubricas/Entrega_2.pdf) |
| **Entrega final** | Secuenciamiento e integración: programación de la semana focal y revisión de la factibilidad del plan | 40 % | [Entrega 3](CaseStudy/Rubricas/Entrega_3.pdf) |

Cada entrega contempla **informe, presentación oral, repositorio de código y coevaluación**. El enunciado establece evaluaciones independientes de cátedra y laboratorio; cada PDF de `CaseStudy/Rubricas/` incluye **ambas pautas de evaluación**.

Consulte el enunciado para las fechas, límites de extensión, requisitos técnicos y reglas de evaluación individual.

---

## 📚 Bibliografía

Entre las referencias citadas en las diapositivas se encuentran:

- *Operations Management: Sustainability and Supply Chain Management* — Heizer, Render y Munson.
- *Operations Management: Processes and Supply Chains* — Krajewski, Malhotra y Ritzman.
- *Introduction to Statistical Quality Control* — Montgomery.
- *Out of the Crisis* — Deming.
- *What Is Total Quality Control? The Japanese Way* — Ishikawa.
- *Measurement Systems Analysis* — Automotive Industry Action Group (AIAG).

Las referencias específicas de cada tema se indican en las presentaciones. Los textos completos de estos libros no forman parte del repositorio.

---

## 📈 Evaluación y registro de notas

La ponderación grupal del caso se calcula como:

$$
N_{\mathrm{caso}} = 0.25\,N_{E1} + 0.35\,N_{E2} + 0.40\,N_{EF}
$$

Esta fórmula se aplica por separado a las notas del caso en cátedra y laboratorio. El enunciado utiliza una **exigencia de 60 %**, equivalente a **nota 4,0** en la escala de 1,0 a 7,0, y detalla los ajustes individuales de cátedra por presentación y coevaluación.

Las ponderaciones anteriores corresponden al **caso integrador**. Consulte las indicaciones del equipo docente para la evaluación global del curso y el acceso al registro de calificaciones, que no forma parte de los archivos de este repositorio.

---

## 📝 Licencia

El archivo [`LICENSE`](LICENSE) establece **[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)** para este repositorio.

Consulte ese archivo para conocer los términos completos de uso y distribución del material.

---

## 👥 Autores

En orden alfabético por apellido:

- **Sebastián Dávila, Ph.D.**
- **Cristian Durán, Ph.D.**
- **Luis Rojo-González, Ph.D.**

Departamento de Ingeniería Industrial · Universidad de Santiago de Chile

---

<div align="center">
<sub>Material docente · Gestión de Operaciones · USACH</sub>
</div>
