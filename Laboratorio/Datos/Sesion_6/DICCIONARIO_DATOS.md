# Laboratorio 6: World Food Programme: asignar recursos escasos

Profesor de laboratorio: Manuel López.

Datos sintéticos docentes. Se conservan los valores numéricos del adjunto; se añaden identificadores, unidades, particiones y parámetros para cargarlos sin transcripción manual.

CSV en UTF-8, separador coma y punto decimal. Encabezados sin tildes. Cada fila representa la entidad descrita; no contiene subtotales ni resultados óptimos.

## datos/proveedores.csv

Tres orígenes potenciales; E no participa en base ni cierre. Filas: 3.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| proveedor | str | identificador | Origen P1, P2 o E. |
| oferta_kits | float | kits | Capacidad máxima de despacho. |
| emergencia | int | indicador 0/1 | 1 únicamente para E; requiere activación en el escenario. |

## datos/centros.csv

Dos centros con demanda obligatoria. Filas: 2.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| centro | str | identificador | Destino C1 o C2. |
| demanda_kits | float | kits | Demanda que debe satisfacerse exactamente. |

## datos/arcos.csv

Seis arcos potenciales; disponibilidad según escenarios.csv. Filas: 6.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| proveedor | str | identificador | Clave de proveedores.csv. |
| centro | str | identificador | Clave de centros.csv. |
| costo_unitario | float | u.m./kit | Costo de envío por kit. |

## datos/escenarios.csv

Tres configuraciones de disponibilidad, no escenarios probabilísticos. Filas: 3.

| Campo | Tipo | Unidad | Definición |
| --- | --- | --- | --- |
| escenario | str | identificador | Orden de análisis: base, cierre, cierre_con_emergencia. |
| habilitar_emergencia | int | indicador 0/1 | 1 activa capacidad de E; 0 fija sus envíos en cero. |
| cerrar_p1_c1 | int | indicador 0/1 | 1 fija x(P1,C1)=0; 0 mantiene el arco disponible. |

## datos/parametros.json

| Clave | Valor | Significado |
| --- | --- | --- |
| laboratorio | 6 | Identificador de sesión. |
| origen_datos | sinteticos_docentes | Procedencia, sin atribución de datos reales a la organización. |
| profesor_laboratorio | Manuel López | Docente responsable del laboratorio. |
| tolerancia | 1e-06 | Tolerancia absoluta para conservación de flujos y cotas. |

## Uso y validación

- Abra un notebook nuevo junto a la guía y copie la celda inicial de lectura.
- Use rutas relativas y conserve la subcarpeta datos.
- Los CSV contienen entradas; el estudiante calcula decisiones, indicadores y gráficos.
- Los archivos tienen conjuntos fijos de datos; no requieren generación aleatoria ni semilla.
- No hay datos ausentes, salvo el plazo del depósito en laboratorio 8: vacío significa no aplicable, no cero.

Referencia de motivación: [Food Assistance Amid Emergency Responses: WFP Awarded the 2021 INFORMS Edelman Award — INFORMS](https://www.informs.org/News-Room/INFORMS-Releases/News-Releases/Food-Assistance-Amid-Emergency-Responses-The-United-Nations-World-Food-Programme-WFP-Awarded-the-2021-INFORMS-Edelman-Award).
