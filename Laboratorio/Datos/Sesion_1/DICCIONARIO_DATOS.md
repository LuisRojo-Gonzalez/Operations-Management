# Laboratorio 1: Toyota: detectar una alteración del proceso

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/subgrupos.csv

24 subgrupos consecutivos de producción. Filas: 24.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| subgrupo | int | orden temporal | Identificador consecutivo de 1 a 24. |
| inspeccionadas | int | piezas | Tamaño del subgrupo; siempre 50. |
| no_conformes | int | piezas | Piezas no conformes, no número de defectos. |
| fase | str | categoría | referencia: 1-20; monitoreo: 21-24. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 1 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| multiplicador_limites | 3 | Número de desviaciones estándar utilizado en los límites. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [Decoding the DNA of the Toyota Production System — Spear y Bowen, HBR](https://hbr.org/1999/09/decoding-the-dna-of-the-toyota-production-system).
