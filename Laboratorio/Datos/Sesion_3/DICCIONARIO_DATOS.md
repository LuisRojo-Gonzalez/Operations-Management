# Laboratorio 3: Zara: pronóstico que alimenta la reposición

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/ajuste.csv

20 períodos para inicialización y ajuste. Filas: 20.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| periodo | int | período | Índice temporal global; no reiniciar al cambiar de archivo. |
| demanda_unidades | int | prendas/período | Demanda observada de indumentaria industrial. |

## datos/validacion.csv

Cuatro períodos para seleccionar hiperparámetros y método. Filas: 4.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| periodo | int | período | Índice temporal global; no reiniciar al cambiar de archivo. |
| demanda_unidades | int | prendas/período | Demanda observada de indumentaria industrial. |

## datos/prueba_final.csv

Cuatro períodos reservados para evaluación final. Filas: 4.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| periodo | int | período | Índice temporal global; no reiniciar al cambiar de archivo. |
| demanda_unidades | int | prendas/período | Demanda observada de indumentaria industrial. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 3 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| estacionalidad | 4 | Número de períodos del ciclo didáctico. |
| ventana_media_movil | 3 | Cantidad de observaciones pasadas de la media móvil. |
| alphas | [0.2, 0.5, 0.8] | Candidatos de suavizamiento, seleccionados solo en validación. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [Research collaboration and math model guided intern through 'fast fashion' world — MIT News](https://news.mit.edu/2006/lfm-correa-1220).
