"""
============================================================
PASO 12: Compilación de outputs finales para el paper
============================================================
Autor: Pipeline Predicción Informalidad Laboral
Fecha: 2026-06

Qué contiene cada output y en qué sección del paper va:
- Tabla 1: Estadísticas descriptivas → Sección 3
- Tabla 2: Comparación de modelos   → Sección 5.1
- Tabla 3: Blinder-Oaxaca           → Sección 5.3
- Tabla 4: DiD COVID-19             → Sección 5.4
- Tabla A1: Codebook ENAHO          → Apéndice A
- Tabla B1: Hiperparámetros óptimos → Apéndice B
- Figuras 1-5 en PDF 300 dpi        → Secciones 3 y 5
- requirements.txt                  → Replicabilidad
- README.md actualizado             → Replicabilidad

Outputs:
- requirements.txt
- README_actualizado.md
- tablas/tabla_A1_codebook.csv  (+.tex)
- tablas/tabla_B1_hiperparametros.csv  (+.tex)
"""

import pickle
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime

from config import ROOT, PROCESSED, MODELS, FIG, TAB
DATOS   = PROCESSED
MODELOS = MODELS
TABLAS  = TAB
FIGS    = FIG
CODIGO  = ROOT / "src"

print("=" * 60)
print("PASO 12: Compilación de Outputs Finales para el Paper")
print("=" * 60)

def load_m(name):
    with open(MODELOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
def load(name):
    with open(DATOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# 12.1 — Verificar todos los outputs generados
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12.1] Verificando outputs generados...")
tablas_esperadas = [
    "tabla1_descriptivas.csv",
    "tabla2_comparacion_modelos.csv",
    "tabla3_oaxaca.csv",
    "tabla4_did.csv",
    "tabla_coeficientes_lasso.csv",
    "importancia_rf.csv",
    "tabla_shap_media.csv",
]
figuras_esperadas = [
    "figura1_tendencias_informalidad.pdf",
    "figura2_composicion_planilla.pdf",
    "figura3a_importancia_rf.pdf",
    "figura3b_shap_summary.pdf",
    "figura4a_shap_dependence_tamano.pdf",
    "figura4b_shap_dependence_edad.pdf",
    "figura5_eventstudy.pdf",
    "figura_roc_comparacion.pdf",
    "figura_calibracion.pdf",
]
modelos_esperados = [
    "modelo_lasso.pkl", "modelo_rf.pkl", "modelo_xgb.pkl", "shap_values.pkl"
]

print("\n  TABLAS:")
for t in tablas_esperadas:
    exists = (TABLAS / t).exists()
    size   = (TABLAS / t).stat().st_size if exists else 0
    print(f"  {'✅' if exists else '❌'} {t} ({size:,} bytes)")

print("\n  FIGURAS:")
for f in figuras_esperadas:
    exists = (FIGS / f).exists()
    size   = (FIGS / f).stat().st_size if exists else 0
    print(f"  {'✅' if exists else '❌'} {f} ({size:,} bytes)")

print("\n  MODELOS:")
for m in modelos_esperados:
    exists = (MODELOS / m).exists()
    size   = (MODELOS / m).stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {m} ({size:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# 12.2 — Tabla A1: Codebook variables ENAHO (Apéndice A)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12.2] Generando Tabla A1: Codebook ENAHO...")
codebook = [
    {"Variable": "AÑO",        "Descripción": "Año de la encuesta (2015–2025)",                        "Valores": "2015–2025",            "Sección": "Capítulo 500"},
    {"Variable": "OCU500_N",   "Descripción": "Condición de actividad (PEA)",                           "Valores": "1=Ocupado, 2=Desocupado, 3=Inactivo, 4=No PEA", "Sección": "P500"},
    {"Variable": "SEXO",       "Descripción": "Sexo del informante",                                    "Valores": "1=Hombre, 2=Mujer",    "Sección": "P207"},
    {"Variable": "EDAD",       "Descripción": "Edad en años cumplidos",                                  "Valores": "14–98",                "Sección": "P208A"},
    {"Variable": "AREA",       "Descripción": "Área de residencia (derivada de ESTRATO)",                "Valores": "1=Urbano (estrato 1–5), 2=Rural (estrato 6–8)", "Sección": "ESTRATO"},
    {"Variable": "REGION",     "Descripción": "Región natural (derivada de DOMINIO)",                    "Valores": "1=Costa, 2=Sierra, 3=Selva", "Sección": "DOMINIO"},
    {"Variable": "DEP",        "Descripción": "Departamento (26 unidades + Lima Metro)",                 "Valores": "1–26",                 "Sección": "UBIGEO"},
    {"Variable": "GEDAD",      "Descripción": "Grupo de edad",                                           "Valores": "1=Joven(14–29), 2=Adulto(30–59), 3=Mayor(60+)", "Sección": "Derivada"},
    {"Variable": "NIVEL_EDU",  "Descripción": "Nivel educativo alcanzado",                               "Valores": "1=Sin nivel, 2=Primaria, 3=Secundaria, 4=Superior técnica, 5=Superior universitaria, 6=Postgrado", "Sección": "P301A"},
    {"Variable": "OCUPINF",    "Descripción": "Empleo informal (2015–2023, INEI)",                       "Valores": "1=Informal, 2=Formal",  "Sección": "OCUPINF"},
    {"Variable": "P511A",      "Descripción": "Tipo de contrato laboral (2024–2025)",                    "Valores": "7=Sin contrato(informal)", "Sección": "P511A"},
    {"Variable": "P510A1",     "Descripción": "RUC del negocio/empresa (2024–2025)",                     "Valores": "3=Sin RUC(informal)",  "Sección": "P510A1"},
    {"Variable": "P510B",      "Descripción": "Libros contables del negocio (2024–2025)",                "Valores": "2=Sin libros(informal)", "Sección": "P510B"},
    {"Variable": "TEI/Y",      "Descripción": "Tasa/variable de empleo informal (variable objetivo Y)", "Valores": "0=Formal, 1=Informal",  "Sección": "Derivada"},
    {"Variable": "FAC500A",    "Descripción": "Factor de expansión muestral (SIEMPRE usar como peso)", "Valores": "Continuo positivo",     "Sección": "FAC500A"},
    {"Variable": "RESIDENTE",  "Descripción": "Residente habitual del hogar",                            "Valores": "1=Sí (P204=1∧P205=2) ó (P204=2∧P206=1)", "Sección": "P204/P205/P206"},
]
import pandas as pd
tabla_a1 = pd.DataFrame(codebook)
tabla_a1.to_csv(TABLAS / "tabla_A1_codebook.csv", index=False)

# LaTeX
latex_a1 = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Tabla A1. Codebook — Variables ENAHO Capítulo 500 (2015–2025)}",
    r"\label{tab:codebook}",
    r"\small",
    r"\begin{tabular}{llp{5cm}l}",
    r"\hline\hline",
    r"Variable & Descripción & Valores & Fuente \\",
    r"\hline",
]
for _, row in tabla_a1.iterrows():
    desc = row['Descripción'].replace("&","\\&").replace("%","\\%")[:60]
    vals = row['Valores'].replace("&","\\&").replace("%","\\%")[:40]
    latex_a1.append(f"\\texttt{{{row['Variable']}}} & {desc} & {vals} & {row['Sección']} \\\\")
latex_a1 += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]
with open(TABLAS / "tabla_A1_codebook.tex", "w") as f:
    f.write("\n".join(latex_a1))
print(f"  ✅ Tabla A1 guardada")

# ══════════════════════════════════════════════════════════════════════════════
# 12.3 — Tabla B1: Hiperparámetros óptimos (Apéndice B)
# ══════════════════════════════════════════════════════════════════════════════
print("[12.3] Generando Tabla B1: Hiperparámetros óptimos...")
try:
    lasso_info = load_m("resultados_lasso")["lasso"]
    rf_info    = load_m("resultados_rf")["rf"]
    xgb_info   = load_m("resultados_xgb")["xgb"]
    
    hiperparams = []
    hiperparams.append({"Modelo": "LASSO Logístico", "Hiperparámetro": "C (= 1/λ)",
                        "Valor": str(lasso_info.get("C_optimo","N/A")), "Método búsqueda": "CV 5-fold (LogisticRegressionCV)"})
    hiperparams.append({"Modelo": "LASSO Logístico", "Hiperparámetro": "penalty",
                        "Valor": "l1 (LASSO)", "Método búsqueda": "Fijo"})
    
    if "params" in rf_info:
        for k, v in rf_info["params"].items():
            hiperparams.append({"Modelo": "Random Forest", "Hiperparámetro": k,
                                "Valor": str(v), "Método búsqueda": "GridSearchCV 5-fold"})
    
    if "params" in xgb_info:
        for k, v in xgb_info["params"].items():
            hiperparams.append({"Modelo": "XGBoost", "Hiperparámetro": k,
                                "Valor": str(v), "Método búsqueda": "RandomizedSearchCV (n=30)"})
    
    tabla_b1 = pd.DataFrame(hiperparams)
    tabla_b1.to_csv(TABLAS / "tabla_B1_hiperparametros.csv", index=False)
    print(f"  ✅ Tabla B1 guardada ({len(hiperparams)} hiperparámetros)")
except Exception as e:
    print(f"  ⚠️ Tabla B1 parcial: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 12.4 — requirements.txt con versiones exactas
# ══════════════════════════════════════════════════════════════════════════════
print("[12.4] Generando requirements.txt...")
paquetes = ["pandas","numpy","scikit-learn","xgboost","shap",
            "matplotlib","statsmodels","seaborn","scipy"]
req_lines = [f"# Pipeline Predicción Informalidad Laboral Perú 2015-2025",
             f"# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"# Python {sys.version}",
             ""]

for pkg in paquetes:
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}; print({pkg}.__version__)"],
            capture_output=True, text=True, timeout=10
        )
        version = result.stdout.strip() if result.returncode == 0 else "unknown"
        req_lines.append(f"{pkg}=={version}")
    except Exception:
        req_lines.append(f"{pkg}")

with open(ROOT / "requirements.txt", "w") as f:
    f.write("\n".join(req_lines))
print(f"  ✅ requirements.txt guardado")
for line in req_lines[3:]:
    if line:
        print(f"     {line}")

# ══════════════════════════════════════════════════════════════════════════════
# 12.5 — Actualizar README.md con instrucciones completas de reproducción
# ══════════════════════════════════════════════════════════════════════════════
print("[12.5] Actualizando README.md...")
readme = f"""# Predicción de Informalidad Laboral en Perú (2015–2025)

> **Artículo empírico para revista Q1** — Journal of Development Economics / World Development  
> Clasificación JEL: J21 · J46 · C55 · O17

## Resumen

Pipeline completo de Machine Learning para predecir la probabilidad de informalidad laboral en Perú, combinando datos de la **ENAHO Capítulo 500** (2015-2025) con la **Planilla Electrónica SUNAT/MTPE**. Se implementan tres modelos predictivos (LASSO Logístico, Random Forest, XGBoost), descomposición Blinder-Oaxaca y análisis de Diferencias en Diferencias con el COVID-19 como shock exógeno.

## Tasa de informalidad estimada
- **73.66%** (promedio ponderado 2015–2025, método TEI compuesto)
- Consistente con estimaciones del INEI (70–75%)

## Estructura de Archivos

```
mtpe/
├── codigo/                  # Scripts Python en orden de ejecución
│   ├── 02_limpieza_planilla.py
│   ├── 03_descriptivos.py
│   ├── 04_features.py
│   ├── 05_lasso.py
│   ├── 06_random_forest.py
│   ├── 07_xgboost_shap.py
│   ├── 08_comparacion_modelos.py
│   ├── 09_oaxaca.py
│   ├── 10_did_covid.py
│   ├── 11_robustez.py
│   └── 12_outputs_paper.py
├── datos/                   # Datasets procesados (generados por los scripts)
│   ├── enaho_empleo_2015_2025_final.csv  ← Ya existe (Paso 1 Python)
│   ├── planilla_limpia.csv
│   ├── dataset_integrado.csv
│   └── *.pkl                # Artefactos de ML (X_train, X_test, etc.)
├── modelos/                 # Modelos entrenados
│   ├── modelo_lasso.pkl
│   ├── modelo_rf.pkl
│   ├── modelo_xgb.pkl
│   └── shap_values.pkl
├── tablas/                  # Tablas en CSV y LaTeX
├── figuras/                 # Figuras en PDF (300 dpi)
├── analisis_enaho/          # Paso 1: Limpieza ENAHO (ya completado)
│   ├── analisis_empleo_enaho_2015_2025.ipynb
│   └── enaho_empleo_2015_2025_final.csv
├── data_empleo/             # Planilla Electrónica SUNAT/MTPE (raw)
├── enaho01a-500/            # Datos crudos ENAHO por año (CSV)
└── requirements.txt

```

## Instrucciones de Reproducción

### 1. Configurar entorno

```bash
conda activate upeu
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline en orden

```bash
# El Paso 1 (Limpieza ENAHO) ya está completo:
# Ver: analisis_enaho/analisis_empleo_enaho_2015_2025.ipynb

# Ejecutar pasos 2-12:
python3 codigo/02_limpieza_planilla.py   # ~2 min
python3 codigo/03_descriptivos.py        # ~3 min
python3 codigo/04_features.py            # ~2 min
python3 codigo/05_lasso.py              # ~10 min
python3 codigo/06_random_forest.py       # ~20 min
python3 codigo/07_xgboost_shap.py        # ~15 min
python3 codigo/08_comparacion_modelos.py # ~2 min
python3 codigo/09_oaxaca.py             # ~5 min
python3 codigo/10_did_covid.py           # ~5 min
python3 codigo/11_robustez.py            # ~15 min
python3 codigo/12_outputs_paper.py       # ~2 min
```

### 3. Verificaciones clave

| Check | Valor esperado | Fuente |
|-------|---------------|--------|
| Tasa informalidad ponderada | 70–75% | INEI |
| AUC-ROC LASSO | 0.74–0.79 | Paso 5 |
| AUC-ROC Random Forest | 0.79–0.84 | Paso 6 |
| AUC-ROC XGBoost | 0.82–0.87 | Paso 7 |
| Pre-tendencias DiD (2015–2019) | β ≈ 0, no significativo | Paso 10 |
| Placebo DiD (año=2017) | β no significativo | Paso 11 |

## Notas Metodológicas

1. **Siempre usar `fac500a`** como peso en estimaciones ENAHO → resultados representativos a nivel nacional.
2. **TEI compuesto** para 2024–2025: se combinan `P511A`, `P510A1`, `P510B` (OR inclusivo) por ausencia de `OCUPINF` en microdatos preliminares.
3. **División temporal**: train = 2015–2021, test = 2022–2025. NO mezclar por data leakage.
4. **Escalado**: StandardScaler SOLO para LASSO. RF y XGBoost son invariantes a escala.
5. **Blinder-Oaxaca**: implementado con WLS (LPM ponderado) equivalente al comando `oaxaca` de Stata.
6. **DiD**: proxy de sector de alto contacto = área urbana (limitación: ENAHO no tiene sector económico a nivel individual).

## Datos

- **ENAHO Capítulo 500**: INEI, disponible en https://iinei.inei.gob.pe/microdatos/
- **Planilla Electrónica**: MTPE, disponible en https://www.gob.pe/mtpe
- Período: 2015–2025 (11 olas anuales ENAHO + 11 años Planilla)

---
*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(ROOT / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print(f"  ✅ README.md actualizado")

# ══════════════════════════════════════════════════════════════════════════════
# 12.6 — Resumen final de todos los outputs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETO — RESUMEN DE OUTPUTS")
print("=" * 60)
print(f"\n📁 Directorio: {ROOT}")
print(f"\n📊 TABLAS ({TABLAS}):")
for f in sorted(TABLAS.glob("*.csv")):
    print(f"   {f.name} ({f.stat().st_size/1024:.0f} KB)")
print(f"\n🖼️  FIGURAS ({FIGS}):")
for f in sorted(FIGS.glob("*.pdf")):
    print(f"   {f.name} ({f.stat().st_size/1024:.0f} KB)")
print(f"\n🤖 MODELOS ({MODELOS}):")
for f in sorted(MODELOS.glob("*.pkl")):
    print(f"   {f.name} ({f.stat().st_size/1e6:.1f} MB)")
print(f"\n📝 CÓDIGO ({CODIGO}):")
for f in sorted(CODIGO.glob("*.py")):
    print(f"   {f.name}")
print(f"\n📄 RAÍZ:")
print(f"   requirements.txt ({(ROOT/'requirements.txt').stat().st_size} bytes)")
print(f"   README.md ({(ROOT/'README.md').stat().st_size/1024:.0f} KB)")
