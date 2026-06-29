# Análisis de Empleo e Informalidad Laboral en el Perú (ENAHO 2015–2025)

Este proyecto realiza un procesamiento, limpieza y análisis longitudinal de la Encuesta Nacional de Hogares (**ENAHO**), específicamente del **Capítulo 500 (Empleo y Ingresos)** para el periodo 2015-2025. El objetivo principal es generar una serie histórica unificada y confiable de indicadores del mercado laboral peruano (PEA, Empleo, Desempleo e Informalidad) adaptada para pipelines de Machine Learning y reportes estadísticos.

---

## 🛠️ Desafíos de Datos Detectados y Soluciones Implementadas

Durante la integración de los 11 años de encuestas (aproximadamente 1 millón de registros en total), se identificaron inconsistencias estructurales en las fuentes de datos del INEI. Se implementó un pipeline en Python (`fix_notebooks.py`) para resolverlas automáticamente:

### 1. Detección Automática de Delimitadores (CSV)
* **Desafío**: La mayoría de los CSV de ENAHO usan coma (`,`) como separador, pero el archivo del año **2025** utiliza punto y coma (`;`). Esto causaba errores de lectura o cargaba registros corruptos (una sola columna con todo el texto concatenado).
* **Solución**: Se implementó una lógica de pre-lectura de la primera línea de cada archivo para contar las ocurrencias de `;` y `,`, seleccionando dinámicamente el separador adecuado (`sep=';'` o `sep=','`) en `pd.read_csv`.

### 2. Estandarización de Factores de Expansión (`FAC500A`)
* **Desafío**: En la base del año **2025**, los números de la columna `FAC500A` (factor de expansión de la submuestra de empleo) venían representados con coma como separador decimal (ej. `150,298` en lugar de `150.298`). Al leerlos como cadena, la conversión directa a flotante fallaba o alteraba los valores de manera masiva.
* **Solución**: Se integró una limpieza que reemplaza `,` por `.` antes de convertir la columna mediante `pd.to_numeric(..., errors='coerce')`, garantizando que la ponderación de la PEA y el empleo sea exacta.

### 3. Ausencia de `OCUPINF` en Microdatos 2024 y 2025 (Tasa de Informalidad)
* **Desafío**: Tradicionalmente, la informalidad laboral se calcula directamente con la columna `OCUPINF` calculada por el INEI. Sin embargo, en los años **2024** y **2025** esta columna no está disponible en los microdatos preliminares/entregados. Un cálculo inicial simplificado reducía drásticamente la tasa de informalidad a ~58% (un sesgo erróneo debido a la omisión de trabajadores informales independientes).
* **Solución**: Se diseñó una lógica de **Índice de Empleo Informal Compuesto (TEI)** utilizando tres variables fundamentales del Capítulo 500:
  1. **Registro en SUNAT / RUC (`P510A1`)**: Identifica si el negocio o empresa del empleador tiene RUC. Si el negocio no cuenta con RUC (valor `'3'`), el empleo se clasifica como **informal**.
  2. **Libros Contables (`P510B`)**: Evalúa si el negocio lleva contabilidad. Si no lleva libros contables (valor `'2'`), se clasifica como **informal**.
  3. **Contrato Laboral (`P511A`)**: Determina el tipo de acuerdo laboral para dependientes. Si el trabajador labora sin contrato escrito (valor `'7'`), se clasifica como **informal**.

  > [!IMPORTANT]
  > **Lógica del Índice Compuesto (OR Inclusivo):** Un trabajador se considera **informal** si al menos uno de los tres indicadores anteriores señala informalidad. Se clasifica como **formal** si cuenta con todas las condiciones formales, y como `NaN` si no aplica al perfil del encuestado (por ejemplo, personas inactivas o menores de edad). Esta reconstrucción metodológica devolvió la informalidad laboral a niveles históricos consistentes (~74%).

---

## 📈 Serie Histórica Nacional (Resultados Clave)

A continuación se detallan los indicadores agregados calculados a nivel nacional (ponderados con `FAC500A`):

| Año | PEA Total | Ocupados | Desocupados | Tasa Actividad (%) | Tasa Desempleo (%) | Tasa Informalidad (TEI %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015** | 16.50 M | 15.92 M | 579 K | 71.63% | 3.51% | **73.15%** |
| **2016** | 16.90 M | 16.20 M | 707 K | 72.23% | 4.18% | **71.97%** |
| **2017** | 17.22 M | 16.51 M | 705 K | 72.42% | 4.09% | **72.55%** |
| **2018** | 17.46 M | 16.78 M | 686 K | 72.33% | 3.93% | **72.44%** |
| **2019** | 17.83 M | 17.13 M | 697 K | 72.74% | 3.91% | **72.74%** |
| **2020** *(Pandemia)* | 16.09 M | 14.90 M | 1.19 M | 64.69% | 7.41% | **75.35%** |
| **2021** | 18.15 M | 17.12 M | 1.03 M | 71.88% | 5.67% | **76.85%** |
| **2022** | 18.55 M | 17.76 M | 795 K | 72.41% | 4.28% | **75.70%** |
| **2023** | 18.63 M | 17.75 M | 879 K | 71.70% | 4.72% | **73.88%** |
| **2024** | 18.82 M | 17.93 M | 893 K | 71.40% | 4.74% | **74.81%** *(Índice Compuesto)* |
| **2025** *(Parcial)* | 19.04 M | 18.15 M | 887 K | 71.24% | 4.66% | **73.85%** *(Índice Compuesto)* |

> [!NOTE]
> Los resultados de 2025 muestran consistencia en la tasa de informalidad (73.85%) y tasas de desempleo (4.66%) tras corregir los decimales del factor de expansión y aplicar la metodología del índice compuesto.

---

## 📁 Estructura de Entregables e Indicadores Exportados

Los datos procesados se exportaron en formato CSV dentro del directorio del proyecto para facilitar análisis posteriores y entrenamiento de modelos predictivos:

1. **`enaho_empleo_2015_2025_final.csv`**: Dataset consolidado y preprocesado a nivel de microdatos (registros individuales con variables estandarizadas). Tamaño: ~126 MB.
2. **`indicadores_nacionales_2015_2025.csv`**: Agregado nacional de la PEA, empleo, desempleo, tasas e informalidad.
3. **`indicadores_area_2015_2025.csv`**: Desglose por Área de Residencia (Urbano vs Rural).
4. **`indicadores_region_2015_2025.csv`**: Desglose por Región Natural (Costa, Sierra, Selva).
5. **`indicadores_sexo_2015_2025.csv`**: Desglose de indicadores por Sexo.
6. **`indicadores_gedad_2015_2025.csv`**: Desglose por Grupos de Edad (Jóvenes, Adultos, Adultos Mayores).
7. **`indicadores_departamento_2015_2025.csv`**: Desglose detallado a nivel de los 24 departamentos y la provincia constitucional del Callao.
8. **`tendencias_estadisticas.csv`**: Resumen analítico de variaciones interanuales y métricas de dispersión.

---

## 📊 Visualizaciones Generadas (`imagenes/`)

Se generaron y exportaron **11 gráficos analíticos** en formato PNG dentro del subdirectorio [imagenes/](file:///home/student2/labs/face/empleo/mtpe/analisis_enaho/imagenes) para reportes de presentación:

* **`series_temporales_nacionales.png`**: Evolución nacional de la Tasa de Informalidad, Desempleo y Actividad.
* **`series_area_residencia.png`**: Brecha de informalidad y desempleo entre el sector Urbano y Rural.
* **`series_region_natural.png`**: Comparación de dinámicas laborales en Costa, Sierra y Selva.
* **`series_por_sexo.png`**: Evolución temporal diferenciada para hombres y mujeres.
* **`series_grupo_edad.png`**: Comportamiento de las tasas por rangos de edad.
* **`heatmap_departamentos.png`**: Mapa de calor que muestra la evolución de la tasa de informalidad por departamento.
* **`series_informalidad_dep.png`**: Trayectoria de la informalidad de los departamentos más representativos.
* **`variacion_anual_barras.png`**: Crecimiento o contracción porcentual interanual de la PEA y Ocupados.
* **`boxplot_distribucion_anual.png`**: Dispersión departamental de la informalidad a lo largo del tiempo.
* **`correlacion_indicadores.png`**: Matriz de correlación lineal entre las variables del mercado de trabajo.
* **`comparacion_2015_vs_2025.png`**: Gráfico comparativo que ilustra los cambios estructurales al inicio y al final de la década de análisis.

---

## 🚀 Entorno de Ejecución y Aceleración por Hardware (GPU)

El procesamiento de los microdatos está optimizado para ejecutarse en entornos con aceleración por hardware:

* **Hardware Utilizado**: Sistema equipado con GPU dedicada (NVIDIA RTX 3080).
* **Framework**: PyTorch (`torch`) para el cálculo matricial acelerado por CUDA de promedios ponderados y agregaciones.
* **Ambiente Conda Recomendado**: `upeu`
* **Comando para re-ejecutar o compilar el notebook**:
  ```bash
  /home/student2/miniconda3/envs/upeu/bin/jupyter-nbconvert --to notebook --execute analisis_empleo_enaho_2015_2025.ipynb --output analisis_empleo_enaho_2015_2025_ejecutado.ipynb
  ```

---

