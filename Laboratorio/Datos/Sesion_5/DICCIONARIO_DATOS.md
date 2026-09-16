# Laboratorio 5: Intel: coordinar producción e inventario

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/periodos.csv

Tres períodos de demanda y capacidad. Filas: 3.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| periodo | int | período | Orden temporal de planificación. |
| demanda_unidades | float | unidades agregadas/período | Demanda que debe atenderse en cada período. |
| capacidad_regular | float | unidades agregadas/período | Límite de producción regular. |
| capacidad_extra | float | unidades agregadas/período | Límite de producción extraordinaria. |

## datos/costos_escenarios.csv

Dos escenarios económicos; mismas demandas y capacidades. Filas: 2.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| escenario | str | identificador | base o sensibilidad. |
| costo_regular | float | u.m./unidad | Costo de producir una unidad regular. |
| costo_extra | float | u.m./unidad | Costo de producir una unidad extraordinaria. |
| costo_inventario | float | u.m./unidad/período | Costo del inventario al cierre de cada período. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 5 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| inventario_inicial | 0 | Unidades agregadas disponibles antes del período 1. |
| inventario_final | 0 | Unidades agregadas exigidas al cierre del período 3. |
| tolerancia | 1e-06 | Tolerancia absoluta para balances y cotas. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [Intel uses O.R. in Corporate Decision-making, Realizing $25 Billion in Benefits — INFORMS](https://www.informs.org/Impact/O.R.-Analytics-Success-Stories/Intel-uses-O.R.-in-Corporate-Decision-making-Realizing-25-Billion-in-Benefits).
