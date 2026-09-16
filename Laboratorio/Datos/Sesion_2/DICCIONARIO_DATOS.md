# Laboratorio 2: Virginia Mason: priorizar mejora y reducir reproceso

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/incidentes.csv

Cuatro categorías de incidentes de mantenimiento. Filas: 4.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| tipo | str | categoría | Tipo de incidente registrado. |
| frecuencia | int | incidentes | Conteo en un mismo período de observación. |
| costo_unitario_um | float | u.m./incidente | Costo didáctico por evento; u.m. es unidad monetaria. |

## datos/proyectos.csv

Tres proyectos de mejora independientes. Filas: 3.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| proyecto | str | identificador | Proyectos indivisibles A, B y C. |
| inversion_millones_clp | float | millones de CLP | Desembolso único del proyecto. |
| beneficio_millones_clp | float | millones de CLP | Beneficio estimado en un horizonte común. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 2 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| presupuesto_millones_clp | 9 | Presupuesto máximo; misma unidad monetaria que las inversiones. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [How Lean Health Care Improves Quality, Values Staff and Reduces Cost — Virginia Mason Institute](https://www.virginiamasoninstitute.org/blog/how-lean-health-care-improves-quality-values-staff-and-reduces-cost).
