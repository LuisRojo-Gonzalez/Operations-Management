# Laboratorio 8: UPS ORION: ordenar entregas con compromisos

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/nodos.csv

Un depósito y cuatro clientes, con plazos de llegada. Filas: 5.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| nodo | int | identificador | 0 es depósito; 1-4 son clientes. |
| tipo | str | categoría | deposito o cliente. |
| limite_llegada_min | float/null | minutos desde salida | Plazo del cliente; vacío en depósito significa no aplicable. |
| servicio_min | float | minutos | Tiempo de servicio; cero para todos. |

## datos/viajes.csv

20 arcos dirigidos; tiempos simétricos. La diagonal se omite y vale cero. Filas: 20.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| origen | int | identificador | Nodo de salida; clave de nodos.csv. |
| destino | int | identificador | Nodo de llegada distinto de origen. |
| tiempo_min | float | minutos | Tiempo del tramo directo, constante durante la ruta. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 8 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| deposito | 0 | ID del nodo de origen y retorno. |
| salida_min | 0 | Instante inicial en minutos. |
| tolerancia | 1e-09 | Tolerancia absoluta para comparar tiempos. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [UPS 2016 Edelman — INFORMS](https://www.informs.org/Recognizing-Excellence/Award-Recipients/UPS-2016-Edelman).
