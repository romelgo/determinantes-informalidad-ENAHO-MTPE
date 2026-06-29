"""
============================================================
PASO 3: Estadísticas descriptivas y visualizaciones
         para la Sección 3 del paper
============================================================


Por qué:
- Las estadísticas descriptivas contextualizan el dataset para el lector
- SIEMPRE se usan fac500a como pesos (resultados representativos a nivel nacional)
- Las figuras deben estar en 300 dpi para cumplir estándares de revistas Q1

Outputs:
- tablas/tabla1_descriptivas.csv  (+ .tex para LaTeX)
- figuras/figura1_tendencias_informalidad.pdf
- figuras/figura2_composicion_planilla.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
from config import INTERIM, RAW_PLANILLA, TAB, FIG
DATA    = INTERIM / "dataset_integrado.csv"
PLAN    = INTERIM / "planilla_limpia.csv"
DATA_EMP = RAW_PLANILLA
TABLAS  = TAB
FIGS    = FIG
TABLAS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ── Paleta de colores (tonos azul/naranja coherentes con estilo Q1) ────────
COL_FORMAL   = "#2563EB"   # Azul
COL_INFORMAL = "#F59E0B"   # Ámbar
COL_URBANO   = "#0891B2"   # Cyan
COL_RURAL    = "#DC2626"   # Rojo

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

print("=" * 60)
print("PASO 3: Análisis Descriptivo")
print("=" * 60)

# ── Cargar dataset integrado ─────────────────────────────────────────────────
df = pd.read_csv(DATA, low_memory=False)
planilla = pd.read_csv(PLAN)
print(f"  Dataset shape: {df.shape}")

# ── Recodificación de variables binarias (INEI usa 1/2, nosotros necesitamos 0/1) ──
# ENAHO original: sexo 1=Hombre, 2=Mujer → recodificar a 0=Hombre, 1=Mujer
# ENAHO original: area 1=Urbano, 2=Rural  → recodificar a 0=Urbano, 1=Rural
# Sin esto, la media de la dummy excede 1, distorsionando coeficientes del modelo.
df["sexo"] = df["sexo"].map({1: 0, 2: 1})  # 0: Hombre, 1: Mujer
df["area"] = df["area"].map({1: 0, 2: 1})  # 0: Urbano,  1: Rural
print("  ✅ Variables binarias recodificadas: sexo (0=Hombre,1=Mujer), area (0=Urbano,1=Rural)")


# ══════════════════════════════════════════════════════════════════════════════
# 3.1 — TABLA 1: Estadísticas descriptivas por estatus (formal vs. informal)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.1] Generando Tabla 1: Estadísticas descriptivas...")

# ── Crear dummies de región y grupo_edad ─────────────────────────────────────
# region y departamento son variables NOMINALES: su promedio no tiene
# interpretación (no existe la "región 1.345"). Lo correcto es mostrar
# la distribución porcentual (% en Costa, % en Sierra, % en Selva).
# grupo_edad también es nominal con 3 categorías; se expande a dummies.
# departamento (código ubigeo) se omite de la Tabla 1: 26 categorías se
# representan mejor en un mapa o gráfico de barras separado.
if "region" in df.columns:
    df["region_costa"]  = (df["region"] == 1).astype(int)  # 1 = Costa
    df["region_sierra"] = (df["region"] == 2).astype(int)  # 2 = Sierra
    df["region_selva"]  = (df["region"] == 3).astype(int)  # 3 = Selva
if "grupo_edad" in df.columns:
    df["grupo_joven"]  = (df["grupo_edad"] == 1).astype(int)  # 14–29
    df["grupo_adulto"] = (df["grupo_edad"] == 2).astype(int)  # 30–59
    df["grupo_mayor"]  = (df["grupo_edad"] == 3).astype(int)  # 60+

# Variables para la Tabla 1
# • Continua/Ordinal:  edad, nivel_edu       → Media ± Desv. (interpretable)
# • Dummies (0/1):     sexo, area, región,   → Media = proporción = % de la subpoblación
#                      grupo_edad
# • EXCLUIDAS:         region (código), departamento (ubigeo), grupo_edad (código)
#                      Se reemplazan por sus dummies para que la media sea interpretable
vars_tabla = {
    "edad":           "Edad (años)",
    "sexo":           "Mujer (=1 si mujer)",
    "area":           "Área rural (=1 si rural)",
    "region_costa":   "Región Costa (=1 si Costa)",
    "region_sierra":  "Región Sierra (=1 si Sierra)",
    "region_selva":   "Región Selva (=1 si Selva)",
    "nivel_edu":      "Nivel educativo (1–6)",
    "grupo_joven":    "Joven 14–29 (=1 si joven)",
    "grupo_adulto":   "Adulto 30–59 (=1 si adulto)",
    "grupo_mayor":    "Mayor 60+ (=1 si mayor)",
}

def stats_ponderadas(subdf: pd.DataFrame, var: str, w: str) -> dict:
    """Calcula media, desv. est., min y max ponderados."""
    vals = subdf[var].dropna()
    pesos = subdf.loc[vals.index, w]
    if len(vals) == 0:
        return {"media": np.nan, "desv": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    media = np.average(vals, weights=pesos)
    varianza = np.average((vals - media) ** 2, weights=pesos)
    return {
        "media": round(media, 3),
        "desv":  round(np.sqrt(varianza), 3),
        "min":   round(vals.min(), 3),
        "max":   round(vals.max(), 3),
        "n":     len(vals),
    }

filas = []
for var, etiqueta in vars_tabla.items():
    if var not in df.columns:
        continue
    s_formal   = stats_ponderadas(df[df["Y"] == 0], var, "fac500a")
    s_informal = stats_ponderadas(df[df["Y"] == 1], var, "fac500a")
    filas.append({
        "Variable":           etiqueta,
        "Media_Formal":       s_formal["media"],
        "Desv_Formal":        s_formal["desv"],
        "Min_Formal":         s_formal["min"],
        "Max_Formal":         s_formal["max"],
        "N_Formal":           s_formal["n"],
        "Media_Informal":     s_informal["media"],
        "Desv_Informal":      s_informal["desv"],
        "Min_Informal":       s_informal["min"],
        "Max_Informal":       s_informal["max"],
        "N_Informal":         s_informal["n"],
    })

tabla1 = pd.DataFrame(filas)
tabla1.to_csv(TABLAS / "tabla1_descriptivas.csv", index=False, encoding="utf-8-sig")

# Exportar en LaTeX
latex_rows = []
latex_rows.append(r"\begin{table}[htbp]")
latex_rows.append(r"\centering")
latex_rows.append(r"\caption{Estadísticas descriptivas por estatus de informalidad (ponderadas con \texttt{fac500a})}")
latex_rows.append(r"\label{tab:descriptivas}")
latex_rows.append(r"\begin{tabular}{lcccccccc}")
latex_rows.append(r"\hline\hline")
latex_rows.append(r" & \multicolumn{4}{c}{Formal (Y=0)} & \multicolumn{4}{c}{Informal (Y=1)} \\")
latex_rows.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
latex_rows.append(r"Variable & Media & Desv. & Mín. & Máx. & Media & Desv. & Mín. & Máx. \\")
latex_rows.append(r"\hline")
for _, row in tabla1.iterrows():
    latex_rows.append(
        f"{row['Variable']} & {row['Media_Formal']} & {row['Desv_Formal']} & "
        f"{row['Min_Formal']} & {row['Max_Formal']} & "
        f"{row['Media_Informal']} & {row['Desv_Informal']} & "
        f"{row['Min_Informal']} & {row['Max_Informal']} \\\\"
    )
latex_rows.append(r"\hline\hline")
latex_rows.append(r"\end{tabular}")
latex_rows.append(r"\end{table}")
with open(TABLAS / "tabla1_descriptivas.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_rows))

print(f"  ✅ Tabla 1 guardada: {TABLAS / 'tabla1_descriptivas.csv'}")
print(tabla1[["Variable","Media_Formal","Media_Informal"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 3.2 — FIGURA 1: Tendencia de informalidad por año y área (urbano/rural)
#        con banda de confianza al 95%
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.2] Generando Figura 1: Tendencias de informalidad 2015–2025...")

# Calcular tasa ponderada y IC 95% por año × área
def tasa_ic(grupo: pd.DataFrame) -> pd.Series:
    """Tasa de informalidad ponderada + IC 95% (bootstrap simple)."""
    w    = grupo["fac500a"]
    y    = grupo["Y"]
    tasa = np.average(y, weights=w)
    # Error estándar aproximado (efecto diseño simplificado)
    n    = len(y)
    se   = np.sqrt(tasa * (1 - tasa) / n)
    return pd.Series({
        "tasa":    round(tasa * 100, 2),
        "ic_low":  round((tasa - 1.96 * se) * 100, 2),
        "ic_high": round((tasa + 1.96 * se) * 100, 2),
        "n":       n,
    })

# Área ya recodificada: 0=Urbano, 1=Rural
df_area = (
    df[df["area"].notna()]
    .groupby(["anio", "area"])
    .apply(tasa_ic, include_groups=False)
    .reset_index()
)
df_nacional = (
    df.groupby("anio")
    .apply(tasa_ic, include_groups=False)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))

# Nacional
ax.plot(df_nacional["anio"], df_nacional["tasa"], color="black",
        linewidth=2, linestyle="--", label="Nacional", zorder=5)

# Urbano (area == 0 tras recodificación)
urb = df_area[df_area["area"] == 0]
ax.plot(urb["anio"], urb["tasa"], color=COL_URBANO, linewidth=2,
        marker="o", markersize=5, label="Urbano")
ax.fill_between(urb["anio"], urb["ic_low"], urb["ic_high"],
                color=COL_URBANO, alpha=0.12)

# Rural (area == 1 tras recodificación)
rur = df_area[df_area["area"] == 1]
ax.plot(rur["anio"], rur["tasa"], color=COL_RURAL, linewidth=2,
        marker="s", markersize=5, label="Rural")
ax.fill_between(rur["anio"], rur["ic_low"], rur["ic_high"],
                color=COL_RURAL, alpha=0.12)

# Línea COVID
ax.axvline(2020, color="gray", linewidth=1, linestyle=":", alpha=0.8)
ax.text(2020.1, ax.get_ylim()[0] + 1, "COVID-19\n(2020)",
        fontsize=8, color="gray", va="bottom")

ax.set_xlabel("Año")
ax.set_ylabel("Tasa de informalidad (%)")
ax.set_title("Figura 1. Evolución de la tasa de informalidad laboral en Perú, 2015–2025\n"
             "(ponderada con fac500a; banda = IC 95%)",
             fontsize=10, pad=12)
ax.set_xticks(range(2015, 2026))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(50, 100)
plt.tight_layout()
plt.savefig(FIGS / "figura1_tendencias_informalidad.pdf")
plt.savefig(FIGS / "figura1_tendencias_informalidad.png", dpi=300)
plt.close()
print(f"  ✅ Figura 1 guardada: {FIGS / 'figura1_tendencias_informalidad.pdf'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.3 — FIGURA 2: Composición Planilla por sector y tamaño empresa
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.3] Generando Figura 2: Composición Planilla Electrónica...")

# Sectores disponibles en planilla
cols_sector_planilla = [
    c for c in planilla.columns
    if c not in ["AÑO", "total_formal", "total_formal_sexo", "pct_mujer_formal",
                 "total_formal_edad", "pct_joven_formal", "total_formal_contrato",
                 "pct_indefinido_formal", "total_formal_empresa", "total_mype",
                 "pct_mype_formal", "total_formal_calif", "pct_no_calificado_formal"]
    and not c.startswith("pct_")
    and "total" not in c
]
# Excluir también columnas de hombre/mujer/joven/adulto/indeterminado
cols_sector_planilla = [c for c in cols_sector_planilla
                        if c not in ["hombre","mujer","no_especificado",
                                     "indeterminado","a_plazo_fijo",
                                     "calificado","no_calificado",
                                     "privado_general","agrario",
                                     "microempresa","pequea_empresa"]]

# Intentar extraer las columnas del sector económico (e_tipo_actividad)
df_act = pd.read_csv(DATA_EMP / "e_tipo_actividad.csv", sep=";", encoding="utf-8-sig")
df_act.columns = (
    df_act.columns.str.strip().str.lower()
    .str.replace(r"\s+", "_", regex=True)
    .str.replace(r"[áéíóúñ]", lambda m: {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n"}[m.group()], regex=True)
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)

col_anio_act = next((c for c in df_act.columns if "a" in c and "o" in c and len(c) <= 4), "ao")
df_act.rename(columns={col_anio_act: "anio"}, inplace=True)
cols_no = [c for c in df_act.columns if "no_" in c or c == "mes" or c == "anio"]
cols_s  = [c for c in df_act.columns if c not in cols_no]

for c in cols_s:
    df_act[c] = (
        df_act[c].astype(str).str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

# Agregar por año y calcular participación
df_act_anual = df_act.groupby("anio")[cols_s].mean().reset_index()
df_act_anual["total"] = df_act_anual[cols_s].sum(axis=1)
for c in cols_s:
    df_act_anual[f"pct_{c}"] = df_act_anual[c] / df_act_anual["total"] * 100

# Nombres de sectores para el gráfico
nombres = {c: c.replace("_y_", " y ").replace("_", " ").title() for c in cols_s}

# Barras apiladas por año
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# — Panel izquierdo: Composición por sector ─────────────────────────────────
pct_cols = [f"pct_{c}" for c in cols_s]
colores_sector = ["#1D4ED8","#0891B2","#16A34A","#CA8A04","#9333EA","#DC2626"]
bottom = np.zeros(len(df_act_anual))
years = df_act_anual["anio"].astype(int).values
for i, (c, col) in enumerate(zip(cols_s, colores_sector)):
    vals = df_act_anual[f"pct_{c}"].fillna(0).values
    axes[0].bar(years, vals, bottom=bottom, color=col, alpha=0.85,
                label=nombres[c], width=0.7)
    bottom += vals

axes[0].set_title("Composición del empleo formal\npor sector económico", fontsize=10)
axes[0].set_xlabel("Año")
axes[0].set_ylabel("Participación (%)")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
axes[0].set_xticks(years)
axes[0].set_xticklabels(years, rotation=45)
axes[0].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.0, 1.0), loc="upper right")

# — Panel derecho: Participación MYPE vs Gran empresa ─────────────────────────
df_emp_r = pd.read_csv(DATA_EMP / "e_tipo_empresa_sector.csv", sep=";", encoding="utf-8-sig")
df_emp_r.columns = (
    df_emp_r.columns.str.strip().str.lower()
    .str.replace(r"\s+", "_", regex=True)
    .str.replace(r"[áéíóúñ]", lambda m: {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n"}[m.group()], regex=True)
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)
col_anio_e = next((c for c in df_emp_r.columns if "a" in c and "o" in c and len(c) <= 4), "ao")
df_emp_r.rename(columns={col_anio_e: "anio"}, inplace=True)
cols_no_e = [c for c in df_emp_r.columns if "no_" in c or c == "mes" or c == "anio"]
cols_e = [c for c in df_emp_r.columns if c not in cols_no_e]
for c in cols_e:
    df_emp_r[c] = (
        df_emp_r[c].astype(str).str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
df_emp_anual = df_emp_r.groupby("anio")[cols_e].mean().reset_index()
df_emp_anual["total"] = df_emp_anual[cols_e].sum(axis=1)
cols_mype2 = [c for c in cols_e if "micro" in c or "pequea" in c or "pequ" in c]
cols_gran  = [c for c in cols_e if c not in cols_mype2]
df_emp_anual["pct_mype"]     = df_emp_anual[cols_mype2].sum(axis=1) / df_emp_anual["total"] * 100 if cols_mype2 else 0
df_emp_anual["pct_gran"]     = df_emp_anual[cols_gran].sum(axis=1) / df_emp_anual["total"] * 100
years_e = df_emp_anual["anio"].astype(int).values
axes[1].bar(years_e, df_emp_anual["pct_gran"].values, color="#2563EB",
            alpha=0.85, label="Gran empresa (privado/agrario)", width=0.7)
if cols_mype2:
    axes[1].bar(years_e, df_emp_anual["pct_mype"].values,
                bottom=df_emp_anual["pct_gran"].values,
                color="#F59E0B", alpha=0.85, label="MYPE", width=0.7)
axes[1].set_title("Composición del empleo formal\npor tamaño de empresa (Ley 30056)", fontsize=10)
axes[1].set_xlabel("Año")
axes[1].set_ylabel("Participación (%)")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
axes[1].set_xticks(years_e)
axes[1].set_xticklabels(years_e, rotation=45)
axes[1].legend(frameon=False, fontsize=8)

fig.suptitle("Figura 2. Empleo formal según Planilla Electrónica SUNAT/MTPE, 2015–2025",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGS / "figura2_composicion_planilla.pdf")
plt.savefig(FIGS / "figura2_composicion_planilla.png", dpi=300)
plt.close()
print(f"  ✅ Figura 2 guardada: {FIGS / 'figura2_composicion_planilla.pdf'}")

print("\n✅ PASO 3 COMPLETADO")
print(f"   → {TABLAS / 'tabla1_descriptivas.csv'}")
print(f"   → {FIGS / 'figura1_tendencias_informalidad.pdf'}")
print(f"   → {FIGS / 'figura2_composicion_planilla.pdf'}")
