"""
============================================================
PASO 4 (ENRIQUECIDO): Ingeniería de features y división del dataset
============================================================

QUÉ CAMBIA RESPECTO A LA VERSIÓN PREVIA Y POR QUÉ
-------------------------------------------------
La versión anterior solo usaba variables demográficas/geográficas
(sexo, edad, área, región, departamento, educación) + agregados de Planilla.
El techo predictivo de ese conjunto es AUC ≈ 0.77 (coherente con la
literatura: Gáfaro et al., 2022, World Development, reportan ~0.75 con
predictores exógenos). Para un clasificador OPERATIVO de focalización
—el objetivo de un venue de Data Mining— incorporamos las variables
ESTRUCTURALES del puesto, presentes en el propio módulo 500 de la ENAHO
para los 11 años pero nunca extraídas:

  • P507  → categoría ocupacional (empleador, asalariado, independiente, TFNR, ...)
  • P512A → tamaño de la empresa (proxy MYPE vs. gran empresa)
  • P506R4→ rama de actividad / sector económico (CIIU rev.4, colapsado a sección)
    OJO: P505R4 es la OCUPACIÓN (CNO-2015), NO la rama de actividad. La rama/sector
    económico es P506R4 ("¿a qué se dedica el negocio/empresa?", CIIU rev.4).
  • P513T → horas trabajadas a la semana
  • D529T → ingreso laboral (log)

CAVEAT METODOLÓGICO (documentado en el paper): categoría ocupacional,
tamaño de empresa y rama de actividad están parcialmente relacionados por
construcción con la definición oficial de informalidad de INEI. Por ello el
paper se enmarca como ejercicio PREDICTIVO/operativo, no causal, y la relación
mecánica se reporta de forma transparente. Las variables literales que definen
el TEI 2024-2025 (P511A contrato, P510A1 RUC, P510B libros) se EXCLUYEN del
vector de predictores para evitar circularidad directa.

ESTRATEGIA DE DATOS
-------------------
- Y autoritativo y demografía limpia se toman del dataset ya validado
  analisis_enaho/enaho_empleo_2015_2025_final.csv (columna TEI = OCUPINF para
  2015-2023, TEI construido para 2024-2025).
- Los predictores fuertes se extraen de los crudos enaho01a-500/ y se fusionan
  por clave (CONGLOME, VIVIENDA, HOGAR, SEXO, EDAD) dentro de cada año
  (match 100%, <0.5% de claves ambiguas que se descartan).

Outputs (datos/):
- X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
- w_train.pkl, w_test.pkl
- X_train_scaled.pkl, X_test_scaled.pkl  (para LASSO)
- feature_names.pkl, scaler.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config import RAW_ENAHO, INTERIM, PROCESSED, ENAHO_FINAL
FINAL = ENAHO_FINAL
RAW   = RAW_ENAHO
PLAN  = INTERIM / "planilla_limpia.csv"
OUT   = PROCESSED
YEARS = list(range(2015, 2026))

print("=" * 64)
print("PASO 4 (ENRIQUECIDO): Ingeniería de Features")
print("=" * 64)

# ──────────────────────────────────────────────────────────────────────────────
# 4.0 — Mapa CIIU rev.4: división (2 primeros dígitos) → sección macro-sector
# ──────────────────────────────────────────────────────────────────────────────
def ciiu_section(code):
    """Colapsa el CIIU de 4 dígitos a una sección macro-sectorial (~17 grupos)."""
    if pd.isna(code):
        return "NA"
    try:
        div = int(str(int(code)).zfill(4)[:2])
    except (ValueError, TypeError):
        return "NA"
    if   1 <= div <= 3:   return "A_agro"
    elif 5 <= div <= 9:   return "B_mineria"
    elif 10 <= div <= 33: return "C_manufactura"
    elif 35 <= div <= 39: return "DE_electr_agua"
    elif 41 <= div <= 43: return "F_construccion"
    elif 45 <= div <= 47: return "G_comercio"
    elif 49 <= div <= 53: return "H_transporte"
    elif 55 <= div <= 56: return "I_aloj_comida"
    elif 58 <= div <= 63: return "J_informacion"
    elif 64 <= div <= 66: return "K_financiero"
    elif div == 68:       return "L_inmobiliario"
    elif 69 <= div <= 82: return "MN_profes_admin"
    elif div == 84:       return "O_adm_publica"
    elif div == 85:       return "P_ensenanza"
    elif 86 <= div <= 88: return "Q_salud"
    elif 90 <= div <= 96: return "RS_arte_otros"
    elif 97 <= div <= 98: return "T_hogares"
    else:                 return "U_otros"

# ──────────────────────────────────────────────────────────────────────────────
# 4.1 — Extraer predictores fuertes de los crudos del módulo 500
# ──────────────────────────────────────────────────────────────────────────────
def extraer_crudo(year):
    f = RAW / f"Enaho01a-{year}-500.csv"
    first = open(f, encoding="latin1", errors="replace").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(f, encoding="latin1", sep=sep, low_memory=False, on_bad_lines="skip")
    df.columns = [c.upper().strip() for c in df.columns]

    # Filtros idénticos a los del dataset final, para alinear la población
    df["P500I"] = df["P500I"].astype(str).str.strip().str.zfill(2)
    df = df[df["P500I"] != "00"]
    for c in ["P204", "P205", "P206", "OCU500", "P208A", "P207"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    res = ((df.P204 == 1) & (df.P205 == 2)) | ((df.P204 == 2) & (df.P206 == 1))
    df = df[res & (df.OCU500 < 3) & (df.P208A.between(14, 98))].copy()

    # Clave de merge
    df["SEXO"] = df["P207"]
    df["EDAD"] = df["P208A"]
    for c in ["CONGLOME", "VIVIENDA", "HOGAR"]:
        df[c] = df[c].astype(str).str.strip()

    # Predictores fuertes
    df["catocup"] = pd.to_numeric(df["P507"],  errors="coerce")        # 1..7
    df["tam_emp"] = pd.to_numeric(df["P512A"], errors="coerce")        # 1..5/6
    df["horas"]   = pd.to_numeric(df["P513T"], errors="coerce")        # h/sem
    ing = pd.to_numeric(df.get("D529T", np.nan), errors="coerce")
    df["log_ing"] = np.log1p(ing.clip(lower=0))
    df["sector"]  = pd.to_numeric(df["P506R4"], errors="coerce").map(ciiu_section)

    key = ["CONGLOME", "VIVIENDA", "HOGAR", "SEXO", "EDAD"]
    cols = key + ["catocup", "tam_emp", "horas", "log_ing", "sector"]
    out = df[cols].drop_duplicates(key, keep=False)   # descarta claves ambiguas
    out["AÑO"] = year
    return out

print("\n[4.1] Extrayendo predictores fuertes de los crudos (módulo 500)...")
ricos = []
for y in YEARS:
    r = extraer_crudo(y)
    ricos.append(r)
    print(f"   {y}: {len(r):,} personas con clave única")
df_ricos = pd.concat(ricos, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4.2 — Cargar dataset final (Y autoritativo + demografía) y fusionar
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4.2] Cargando dataset final y fusionando...")
fin = pd.read_csv(FINAL, low_memory=False)
fin.columns = [c.strip().upper().replace("﻿", "") for c in fin.columns]
fin.rename(columns={"DEP": "DEPARTAMENTO"}, inplace=True)

# Y autoritativo: TEI (= OCUPINF 2015-2023, TEI construido 2024-2025)
fin["Y"] = pd.to_numeric(fin["TEI"], errors="coerce")
fin = fin[fin["Y"].isin([0, 1])].copy()

for c in ["CONGLOME", "VIVIENDA", "HOGAR"]:
    fin[c] = fin[c].astype(str).str.strip()
fin["SEXO"] = pd.to_numeric(fin["SEXO"], errors="coerce")
fin["EDAD"] = pd.to_numeric(fin["EDAD"], errors="coerce")

key = ["AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "SEXO", "EDAD"]
df = fin.merge(df_ricos, on=key, how="inner")
print(f"   Filas tras merge: {len(df):,} (de {len(fin):,} en final)")

# grupo_edad consistente con el paper: joven 14-29, adulto 30-59, mayor 60+
df["grupo_edad"] = np.where(df.EDAD.between(14, 29), 1,
                    np.where(df.EDAD.between(30, 59), 2, 3))

# ──────────────────────────────────────────────────────────────────────────────
# 4.3 — Variables de contexto del mercado formal (Planilla, nivel año)
# ──────────────────────────────────────────────────────────────────────────────
plan = pd.read_csv(PLAN)
plan.columns = [c.strip().upper().replace("﻿", "") for c in plan.columns]
plan_vars = ["PCT_MUJER_FORMAL", "PCT_JOVEN_FORMAL", "PCT_INDEFINIDO_FORMAL",
             "PCT_MYPE_FORMAL", "PCT_NO_CALIFICADO_FORMAL"]
plan_vars = [v for v in plan_vars if v in plan.columns]
plan_small = plan[["AÑO"] + plan_vars].rename(columns={v: v.lower() for v in plan_vars})
df = df.merge(plan_small, on="AÑO", how="left")

# ──────────────────────────────────────────────────────────────────────────────
# 4.4 — Construcción del vector X
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4.4] Construyendo vector de features...")

# Recodificar binarias a 0/1
df["sexo"] = df["SEXO"].map({1: 0, 2: 1})              # 0=Hombre, 1=Mujer
df["area"] = pd.to_numeric(df["AREA"], errors="coerce").map({1: 0, 2: 1})  # 0=Urbano,1=Rural
df["edad"] = df["EDAD"].astype(float)
df["region"] = pd.to_numeric(df["REGION"], errors="coerce")
df["departamento"] = pd.to_numeric(df["DEPARTAMENTO"], errors="coerce")
df["nivel_edu"] = pd.to_numeric(df["NIVEL_EDU"], errors="coerce")
df["anio"] = df["AÑO"].astype(int)
df["anio_covid"] = df["anio"].isin([2020, 2021]).astype(int)

# Derivadas de los predictores fuertes
df["mype"] = (df["tam_emp"] <= 1).astype(float)        # empresa pequeña (~MYPE)
df["horas"] = df["horas"].fillna(df["horas"].median())
df["log_ing"] = df["log_ing"].fillna(0.0)

num_feats = ["sexo", "area", "edad", "anio_covid", "horas", "log_ing", "mype"] + \
            [v.lower() for v in plan_vars]

cat_feats = ["grupo_edad", "region", "departamento", "nivel_edu",
             "catocup", "tam_emp", "sector", "anio"]

# cat_feats ya incluye 'anio'; no duplicar
base = df[num_feats + cat_feats + ["Y", "FAC500A"]].copy()
base = base.rename(columns={"FAC500A": "fac500a"})
base["fac500a"] = pd.to_numeric(base["fac500a"], errors="coerce")
base = base.dropna(subset=["Y", "fac500a"]).reset_index(drop=True)

anio_orig = base["anio"].astype(int).values   # capturar antes del encoding

# One-hot encoding de categóricas (drop_first para evitar multicolinealidad)
df_enc = pd.get_dummies(base, columns=cat_feats, drop_first=True, dtype=float)

y = base["Y"].astype(int).reset_index(drop=True)
w = base["fac500a"].reset_index(drop=True)

feature_cols = [c for c in df_enc.columns if c not in ["Y", "fac500a", "anio"]]
X = df_enc[feature_cols].reset_index(drop=True)
print(f"   X shape: {X.shape}")
print(f"   y: formal={(y==0).sum():,} ({(y==0).mean()*100:.1f}%), "
      f"informal={(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# 4.5 — División temporal y filtro de varianza cero (en train)
# ──────────────────────────────────────────────────────────────────────────────
train_mask = anio_orig <= 2021
test_mask  = anio_orig >= 2022

X_train = X[train_mask].reset_index(drop=True)
X_test  = X[test_mask].reset_index(drop=True)
# eliminar columnas constantes en train (p.ej. dummies de años de test)
keep = [c for c in X_train.columns if X_train[c].fillna(0).var() > 0]
dropped = [c for c in X_train.columns if c not in keep]
if dropped:
    print(f"   Columnas con varianza 0 en train eliminadas: {len(dropped)} -> {dropped}")
X_train, X_test = X_train[keep], X_test[keep]
feature_cols = keep

y_train = y[train_mask].reset_index(drop=True)
y_test  = y[test_mask].reset_index(drop=True)
w_train = w[train_mask].reset_index(drop=True)
w_test  = w[test_mask].reset_index(drop=True)

print(f"\n   Train (2015-2021): {len(X_train):,} obs ({y_train.mean()*100:.1f}% informal)")
print(f"   Test  (2022-2025): {len(X_test):,} obs ({y_test.mean()*100:.1f}% informal)")

# ──────────────────────────────────────────────────────────────────────────────
# 4.6 — Escalado (SOLO para LASSO)
# ──────────────────────────────────────────────────────────────────────────────
vars_escalar = [c for c in ["edad", "horas", "log_ing"] + [v.lower() for v in plan_vars]
                if c in X_train.columns]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[vars_escalar] = scaler.fit_transform(X_train[vars_escalar].fillna(0))
X_test_scaled[vars_escalar]  = scaler.transform(X_test[vars_escalar].fillna(0))
print(f"   Variables escaladas: {vars_escalar}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.7 — Guardar artefactos
# ──────────────────────────────────────────────────────────────────────────────
# Año por fila, alineado con X_train / X_test (para análisis por periodo en src/14)
anio_train = pd.Series(anio_orig[train_mask]).reset_index(drop=True)
anio_test  = pd.Series(anio_orig[test_mask]).reset_index(drop=True)

artifacts = {
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "w_train": w_train, "w_test": w_test,
    "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
    "scaler": scaler, "feature_names": list(feature_cols),
    "anio_train": anio_train, "anio_test": anio_test,
}
for name, obj in artifacts.items():
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

print(f"\n   Features totales: {len(feature_cols)}")
print("\n✅ PASO 4 (ENRIQUECIDO) COMPLETADO — artefactos en datos/")
