# Laboratorio 7: Lenovo: comparar reglas con una solución de referencia

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/trabajos.csv

Seis trabajos de una única máquina. Filas: 6.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| trabajo | str | identificador | Trabajo A-F; clave de desempate alfabético. |
| procesamiento_h | float | horas | Tiempo de procesamiento sin interrupción. |
| fecha_entrega_h | float | horas desde t=0 | Fecha comprometida de terminación. |
| familia | str | categoría | Familia X o Y, determina preparación. |
| liberacion_h | float | horas desde t=0 | Disponibilidad inicial; cero para todos. |

## datos/preparaciones.csv

Matriz completa de cuatro transiciones entre familias. Filas: 4.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| familia_origen | str | categoría | Familia del trabajo anterior. |
| familia_destino | str | categoría | Familia del trabajo siguiente. |
| preparacion_h | float | horas | Tiempo consumido en la misma máquina antes del siguiente trabajo. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 7 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| preparacion_inicial_h | 0 | Preparación anterior al primer trabajo; horas. |
| tiempo_inicial_h | 0 | Origen temporal común; horas. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [Lenovo Schedules Laptop Manufacturing Using Deep Reinforcement Learning — Liang y colaboradores, INFORMS Journal on Applied Analytics](https://pubsonline.informs.org/doi/abs/10.1287/inte.2021.1109?journalCode=ijaa).
