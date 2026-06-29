# Determinantes de la informalidad laboral en Perú (2015–2025)

Pipeline reproducible que identifica los **factores con mayor peso** detrás de la informalidad
laboral en Perú y cómo **cambian a lo largo de 2015–2025** (incluido el shock COVID-19), para
focalizar políticas de formalización.

## Fuentes de datos

| Fuente | Descripción | Nivel | Ubicación |
|---|---|---|---|
| **ENAHO** (INEI), Módulo 500 | Encuesta de hogares, empleo e ingresos, 2015–2025 | Individuo (≥14 años) | `data/raw/enaho01a-500/` |
| **Planilla Electrónica** (SUNAT/MTPE) | Trabajadores formales del sector privado dependiente, 2015–2025 | Agregado nacional mensual | `data/raw/planilla/` |

> Nota: los datos crudos (2.5 GB) **no se versionan en git** (ver `.gitignore`); deben colocarse
> manualmente en `data/raw/`.

## Variable objetivo

`TEI` (Tasa de Empleo Informal), binaria: `1 = informal`, `0 = formal`.
- **2015–2023**: `OCUPINF` de INEI (definición oficial).
- **2024–2025**: índice compuesto (sin RUC `P510A1`, sin libros contables `P510B`, sin contrato
  `P511A`) — aproximación documentada como no idéntica a la definición INEI.

## Estructura del repo

```
config.py … en src/      Rutas centralizadas (importadas por todos los scripts)
data/raw/                ENAHO crudo + Planilla (gitignored)
data/interim/            CSV intermedios: final ENAHO, planilla_limpia, integrado (gitignored)
data/processed/          .pkl train/test/scaler (gitignored)
notebooks/               00_eda_enaho · 01_eda_contraste · eda_planilla/ (6 EDA Planilla)
src/                     Pipeline numerado 02→14
models/                  Modelos entrenados (gitignored)
outputs/tables/          Tablas CSV/TEX (versionadas)
outputs/figures/         Figuras (gitignored, regenerables)

```

## Cómo ejecutar

```bash
pip install -r requirements.txt

# EDA (notebooks):  notebooks/00_eda_enaho.ipynb  y  notebooks/01_eda_contraste.ipynb

# Pipeline de modelado (desde la raíz del repo):
python src/02_limpieza_planilla.py     # integra Planilla ↔ ENAHO por año
python src/04_features.py              # features enriquecidas + split temporal 2015-21 / 2022-25
python src/05_lasso.py                 # LASSO logístico
python src/06_random_forest.py         # Random Forest + importancia
python src/07_xgboost_shap.py          # XGBoost + SHAP
python src/08_comparacion_modelos.py   # comparación de modelos (AUC, ROC, calibración)
python src/09_oaxaca.py                # descomposición Blinder-Oaxaca (brecha de género)
python src/10_did_covid.py             # Difference-in-Differences shock COVID
python src/14_determinantes_temporal.py  # ranking consolidado + SHAP por periodo (entregable)
```

## Determinantes principales (resultado)

Los predictores estructurales del módulo 500 (categoría ocupacional `P507`, tamaño de empresa
`P512A`, rama de actividad `P506R4`, horas e ingreso laboral) elevan el AUC de ~0.77 (solo
demografía/geografía) a ~0.96. **Caveat**: estas variables se relacionan parcialmente por
construcción con la definición de informalidad, por lo que el ejercicio es **predictivo/operativo,
no causal**; los proxies literales del TEI (`P511A`, `P510A1`, `P510B`) se excluyen del vector de
predictores. La lectura causal del shock COVID se hace vía Difference-in-Differences
(`src/10_did_covid.py`).
