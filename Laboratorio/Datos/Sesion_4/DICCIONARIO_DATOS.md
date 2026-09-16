# Laboratorio 4: Sport Obermeyer: decidir bajo demanda incierta

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/demanda_escenarios.csv

Tres escenarios mutuamente excluyentes y exhaustivos. Filas: 3.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| escenario | str | identificador | Escenario de demanda bajo, central o alto. |
| demanda_unidades | int | prendas | Demanda total de la temporada. |
| probabilidad | float | proporción | Probabilidad del escenario; las tres suman 1. |

## datos/costos_sensibilidad.csv

Cuatro combinaciones de costos con idénticas probabilidades. Filas: 4.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| caso | str | identificador | base o sensibilidad del costo de faltante. |
| costo_faltante_um | float | u.m./prenda | Costo marginal de demanda no atendida. |
| costo_sobrante_um | float | u.m./prenda | Costo marginal del excedente. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 4 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| q_min | 0 | Lote mínimo, en prendas. |
| q_max | 160 | Lote máximo incluido, en prendas. |
| q_paso | 1 | Paso entero de la enumeración. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [Making Supply Meet Demand in an Uncertain World — Fisher, Hammond, Obermeyer y Raman, HBR](https://hbr.org/1994/05/making-supply-meet-demand-in-an-uncertain-world).
