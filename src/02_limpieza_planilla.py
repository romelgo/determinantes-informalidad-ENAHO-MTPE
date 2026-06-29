"""
============================================================
PASO 2: Limpieza y estandarización de Planilla Electrónica
         + Integración con ENAHO por celdas sintéticas
============================================================

w
Fuente: 6 CSVs de data_empleo/ (datos Planilla Electrónica SUNAT/MTPE)
        Cobertura: 2015-2025, separados por dimensión estructural
Output: planilla_limpia.csv
        dataset_integrado.csv  ← listo para ingeniería de features
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
from config import INTERIM, RAW_PLANILLA, ENAHO_FINAL
DATA_EMP    = RAW_PLANILLA
ENAHO_CSV   = ENAHO_FINAL
OUT_DIR     = INTERIM
OUT_DIR.mkdir(exist_ok=True)

# ── Meses en español → número (para agrupar por año) ─────────────────────────
MESES = {
    "ENE.": 1, "FEB.": 2, "MAR.": 3, "ABR.": 4, "MAY.": 5, "JUN.": 6,
    "JUL.": 7, "AGO.": 8, "SET.": 9, "OCT.": 10, "NOV.": 11, "DIC.": 12,
}

def limpiar_numero(serie: pd.Series) -> pd.Series:
    """Convierte '2,457,994' → 2457994.0 manejando comas como separador de miles."""
    return (
        serie.astype(str)
             .str.replace(",", "", regex=False)
             .pipe(pd.to_numeric, errors="coerce")
    )

def cargar_planilla(filepath: Path, sep: str = ";") -> pd.DataFrame:
    """Carga un CSV de Planilla con BOM y separador ';', normaliza mes y año."""
    df = pd.read_csv(filepath, sep=sep, encoding="utf-8-sig")
    # Estandarizar nombres de columna: minúsculas, sin espacios
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[áéíóú]", lambda m: {"á":"a","é":"e","í":"i","ó":"o","ú":"u"}[m.group()], regex=True)
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    # Estandarizar columna mes y año
    if "mes" in df.columns:
        df["mes_num"] = df["mes"].str.strip().map(MESES)
    # Buscar la columna de año (puede venir como 'año' o 'ano')
    col_anio = next((c for c in df.columns if "a" in c and "o" in c and len(c) <= 4), None)
    if col_anio and col_anio != "anio":
        df.rename(columns={col_anio: "anio"}, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.1 — Cargar y limpiar cada dimensión de Planilla
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PASO 2: Limpieza Planilla Electrónica")
print("=" * 60)

# ── 2.1a: Por sector económico (actividad) ────────────────────────────────────
# Columnas: Agropecuario y pesca, Manufactura, Comercio, Construcción, Minería, Servicios
df_sector = cargar_planilla(DATA_EMP / "e_tipo_actividad.csv")
print(f"\n[Sector] Columnas: {list(df_sector.columns)}")
# Identificar columnas de sectores (todas excepto anio, mes, mes_num, no_especificado)
cols_meta   = ["anio", "mes", "mes_num"]
cols_no_esp = [c for c in df_sector.columns if "no_" in c or "no" == c[:2]]
cols_sector = [c for c in df_sector.columns if c not in cols_meta + cols_no_esp]
for c in cols_sector:
    df_sector[c] = limpiar_numero(df_sector[c])
# Agregar por año (promedio mensual de trabajadores formales por sector)
sector_anual = (
    df_sector.groupby("anio")[cols_sector]
             .mean()
             .reset_index()
)
# Calcular totales y participación porcentual
sector_anual["total_formal"] = sector_anual[cols_sector].sum(axis=1)
for c in cols_sector:
    sector_anual[f"pct_{c}"] = sector_anual[c] / sector_anual["total_formal"]
print(f"[Sector] Dataset anual shape: {sector_anual.shape}")

# ── 2.1b: Por sexo ────────────────────────────────────────────────────────────
df_sexo = cargar_planilla(DATA_EMP / "e_por_sexo.csv")
cols_no_esp_s = [c for c in df_sexo.columns if "no_" in c or c == "no"]
cols_sexo = [c for c in df_sexo.columns if c not in ["anio","mes","mes_num"] + cols_no_esp_s]
for c in cols_sexo:
    df_sexo[c] = limpiar_numero(df_sexo[c])
sexo_anual = df_sexo.groupby("anio")[cols_sexo].mean().reset_index()
sexo_anual["total_formal_sexo"] = sexo_anual[cols_sexo].sum(axis=1)
# Participación femenina en empleo formal
col_mujer = next((c for c in cols_sexo if "mujer" in c or "fem" in c), cols_sexo[-1])
sexo_anual["pct_mujer_formal"] = sexo_anual[col_mujer] / sexo_anual["total_formal_sexo"]
print(f"[Sexo]   Dataset anual shape: {sexo_anual.shape} | col mujer: {col_mujer}")

# ── 2.1c: Por grupo de edad ───────────────────────────────────────────────────
df_edad = cargar_planilla(DATA_EMP / "e_por_edad.csv")
cols_no_esp_e = [c for c in df_edad.columns if "no_" in c or c == "no"]
cols_edad = [c for c in df_edad.columns if c not in ["anio","mes","mes_num"] + cols_no_esp_e]
for c in cols_edad:
    df_edad[c] = limpiar_numero(df_edad[c])
edad_anual = df_edad.groupby("anio")[cols_edad].mean().reset_index()
edad_anual["total_formal_edad"] = edad_anual[cols_edad].sum(axis=1)
col_joven = next((c for c in cols_edad if "jov" in c or "29" in c), cols_edad[0])
edad_anual["pct_joven_formal"] = edad_anual[col_joven] / edad_anual["total_formal_edad"]
print(f"[Edad]   Dataset anual shape: {edad_anual.shape} | col joven: {col_joven}")

# ── 2.1d: Por tipo de contrato ────────────────────────────────────────────────
df_contrato = cargar_planilla(DATA_EMP / "e_tipo_contrato.csv")
cols_no_esp_c = [c for c in df_contrato.columns if "no_" in c or c == "no"]
cols_contrato = [c for c in df_contrato.columns if c not in ["anio","mes","mes_num"] + cols_no_esp_c]
for c in cols_contrato:
    df_contrato[c] = limpiar_numero(df_contrato[c])
contrato_anual = df_contrato.groupby("anio")[cols_contrato].mean().reset_index()
contrato_anual["total_formal_contrato"] = contrato_anual[cols_contrato].sum(axis=1)
col_indetermina = next((c for c in cols_contrato if "indet" in c or "indeter" in c), cols_contrato[0])
contrato_anual["pct_indefinido_formal"] = contrato_anual[col_indetermina] / contrato_anual["total_formal_contrato"]
print(f"[Contrato] Dataset anual shape: {contrato_anual.shape} | col indef: {col_indetermina}")

# ── 2.1e: Por tipo de empresa y sector ───────────────────────────────────────
df_empresa = cargar_planilla(DATA_EMP / "e_tipo_empresa_sector.csv")
cols_no_esp_emp = [c for c in df_empresa.columns if "no_" in c or c == "no"]
cols_empresa = [c for c in df_empresa.columns if c not in ["anio","mes","mes_num"] + cols_no_esp_emp]
for c in cols_empresa:
    df_empresa[c] = limpiar_numero(df_empresa[c])
empresa_anual = df_empresa.groupby("anio")[cols_empresa].mean().reset_index()
empresa_anual["total_formal_empresa"] = empresa_anual[cols_empresa].sum(axis=1)
# Identificar MYPE = microempresa + pequeña empresa
cols_mype = [c for c in cols_empresa if "micro" in c or "peque" in c or "pequea" in c]
if cols_mype:
    empresa_anual["total_mype"] = empresa_anual[cols_mype].sum(axis=1)
    empresa_anual["pct_mype_formal"] = empresa_anual["total_mype"] / empresa_anual["total_formal_empresa"]
print(f"[Empresa] Dataset anual shape: {empresa_anual.shape} | cols MYPE: {cols_mype}")

# ── 2.1f: Por calificación ────────────────────────────────────────────────────
df_calif = cargar_planilla(DATA_EMP / "e_por_calificacion.csv")
cols_no_esp_ca = [c for c in df_calif.columns if "no_" in c or c == "no"]
cols_calif = [c for c in df_calif.columns if c not in ["anio","mes","mes_num"] + cols_no_esp_ca]
for c in cols_calif:
    df_calif[c] = limpiar_numero(df_calif[c])
calif_anual = df_calif.groupby("anio")[cols_calif].mean().reset_index()
calif_anual["total_formal_calif"] = calif_anual[cols_calif].sum(axis=1)
col_no_calif = next((c for c in cols_calif if "no_cal" in c or "nocal" in c), cols_calif[-1])
calif_anual["pct_no_calificado_formal"] = calif_anual[col_no_calif] / calif_anual["total_formal_calif"]
print(f"[Calif]  Dataset anual shape: {calif_anual.shape} | col no-calif: {col_no_calif}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.2 — Construir dataset de Planilla limpio (a nivel año)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2.2] Integrando dimensiones de Planilla por año...")
planilla = sector_anual.copy()
planilla = planilla.merge(sexo_anual[["anio","total_formal_sexo","pct_mujer_formal"]], on="anio", how="outer")
planilla = planilla.merge(edad_anual[["anio","total_formal_edad","pct_joven_formal"]], on="anio", how="outer")
planilla = planilla.merge(contrato_anual[["anio","total_formal_contrato","pct_indefinido_formal"]], on="anio", how="outer")
planilla = planilla.merge(empresa_anual[["anio","total_formal_empresa","pct_mype_formal"] if "pct_mype_formal" in empresa_anual.columns else ["anio","total_formal_empresa"]], on="anio", how="outer")
planilla = planilla.merge(calif_anual[["anio","total_formal_calif","pct_no_calificado_formal"]], on="anio", how="outer")
planilla.rename(columns={"anio": "AÑO"}, inplace=True)

print(f"[Planilla limpia] Shape: {planilla.shape}")
print(f"  Años: {sorted(planilla['AÑO'].dropna().astype(int).unique())}")
print(planilla[["AÑO","total_formal","pct_mujer_formal","pct_joven_formal","pct_indefinido_formal"]].to_string(index=False))

planilla.to_csv(OUT_DIR / "planilla_limpia.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Guardado: {OUT_DIR / 'planilla_limpia.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.3 — Integrar con ENAHO (merge por año)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2.3] Cargando ENAHO y realizando merge por año...")
enaho = pd.read_csv(ENAHO_CSV)
print(f"  ENAHO shape total: {enaho.shape}")

# Filtrar: ocupados residentes con TEI válido (PEA ocupada)
enaho_filtrado = enaho[
    (enaho["OCUPADO"] == 1) &
    (enaho["RESIDENTE"] == 1) &
    enaho["TEI"].notna()
].copy()
print(f"  ENAHO ocupados+residentes+TEI: {enaho_filtrado.shape}")

# Renombrar para consistencia con el prompt (minúsculas)
enaho_filtrado = enaho_filtrado.rename(columns={
    "AÑO":           "anio",
    "SEXO":          "sexo",
    "EDAD":          "edad",
    "AREA":          "area",
    "AREA_LABEL":    "area_label",
    "REGION":        "region",
    "REGION_LABEL":  "region_label",
    "DEP":           "departamento",
    "DEP_LABEL":     "dep_label",
    "GEDAD":         "grupo_edad",
    "GEDAD_LABEL":   "gedad_label",
    "NIVEL_EDU":     "nivel_edu",
    "TEI":           "Y",         # Variable objetivo: 1=informal, 0=formal
    "FAC500A":       "fac500a",
    "OCU500_N":      "ocu500",
})
# Preparar columna año de Planilla (renombrar)
planilla_merge = planilla.rename(columns={"AÑO": "anio"})

# Merge por año (left join para conservar todos los registros ENAHO)
n_antes = len(enaho_filtrado)
dataset = enaho_filtrado.merge(planilla_merge, on="anio", how="left")
n_match = dataset["total_formal"].notna().sum()
pct_match = n_match / n_antes * 100
print(f"  Merge completado: {n_match:,} de {n_antes:,} filas con Planilla ({pct_match:.1f}%)")

# ── Verificaciones ──────────────────────────────────────────────────────────
print("\n── Verificaciones ──────────────────────────────────────────────────")
print(f"  Shape dataset integrado: {dataset.shape}")
print(f"  Años cubiertos: {sorted(dataset['anio'].unique())}")
inf_rate = (dataset["Y"] * dataset["fac500a"]).sum() / dataset["fac500a"].sum() * 100
print(f"  Tasa informalidad ponderada: {inf_rate:.2f}% (esperado: 70–75%)")
print(f"  Y missing: {dataset['Y'].isna().sum()}")
print(f"  fac500a missing: {dataset['fac500a'].isna().sum()}")
print(f"  sexo missing: {dataset['sexo'].isna().sum()}")

# ── Guardar ─────────────────────────────────────────────────────────────────
dataset.to_csv(OUT_DIR / "dataset_integrado.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Guardado: {OUT_DIR / 'dataset_integrado.csv'}")
print(f"   Tamaño: {(OUT_DIR / 'dataset_integrado.csv').stat().st_size / 1e6:.1f} MB")
print("\n✅ PASO 2 COMPLETADO")
