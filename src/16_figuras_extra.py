"""
============================================================
PASO 16: Figuras adicionales 
============================================================


Salidas (outputs/figures/):
  Descriptivas
    series_por_sexo.png
    series_grupo_edad.png
    series_area_residencia.png
    series_region_natural.png
    comparacion_2015_vs_2025.png
    correlacion_indicadores.png
  Robustez (desde CSV)
    figura_robustez_expanding_window.png
    figura_robustez_shap_bootstrap.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import INTERIM, TAB, FIG

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

AZUL, ROJO, VERDE, NARANJA = "#2563EB", "#DC2626", "#16A34A", "#F59E0B"

print("=" * 60)
print("PASO 16: Figuras adicionales (descriptivas + robustez)")
print("=" * 60)

# ── Datos de trabajadores ────────────────────────────────────────────────────
cols = ["anio", "area", "region", "sexo", "grupo_edad", "edad", "nivel_edu",
        "Y", "fac500a", "OCUPADO", "RESIDENTE"]
df = pd.read_csv(INTERIM / "dataset_integrado.csv", usecols=cols, low_memory=False)
df = df[(df["OCUPADO"] == 1) & (df["RESIDENTE"] == 1) & df["Y"].isin([0, 1])].copy()
print(f"  Trabajadores-año: {len(df):,}")


def tasa_ponderada(d):
    """Tasa de informalidad ponderada por fac500a (en %)."""
    w = d["fac500a"]
    return 100.0 * np.average(d["Y"], weights=w)


def serie_por_grupo(col, etiquetas, colores, titulo, archivo, leyenda_titulo):
    """Serie temporal de la tasa de informalidad por categoría de `col`."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for val, (lab, color) in etiquetas.items():
        sub = df[df[col] == val]
        serie = sub.groupby("anio").apply(tasa_ponderada)
        ax.plot(serie.index, serie.values, marker="o", linewidth=2.2,
                color=color, label=lab)
    ax.axvspan(2019.5, 2021.5, color="gray", alpha=0.12, label="COVID-19")
    ax.set_xlabel("Año"); ax.set_ylabel("Tasa de informalidad (%)")
    ax.set_title(titulo, fontsize=12)
    ax.legend(title=leyenda_titulo, frameon=False)
    ax.set_xticks(range(2015, 2026, 2))
    fig.savefig(FIG / archivo)
    plt.close(fig)
    print(f"  → {archivo}")


# 1. Por sexo
serie_por_grupo(
    "sexo", {1: ("Hombre", AZUL), 2: ("Mujer", ROJO)},
    None, "Informalidad laboral por sexo, 2015–2025",
    "series_por_sexo.png", "Sexo")

# 2. Por grupo de edad
serie_por_grupo(
    "grupo_edad", {1: ("14–24 años", AZUL), 2: ("25–44 años", VERDE),
                   3: ("45+ años", NARANJA)},
    None, "Informalidad laboral por grupo de edad, 2015–2025",
    "series_grupo_edad.png", "Grupo de edad")

# 3. Por área
serie_por_grupo(
    "area", {1: ("Urbano", AZUL), 2: ("Rural", VERDE)},
    None, "Informalidad laboral por área de residencia, 2015–2025",
    "series_area_residencia.png", "Área")

# 4. Por región natural
serie_por_grupo(
    "region", {1: ("Costa", AZUL), 2: ("Sierra", NARANJA), 3: ("Selva", VERDE)},
    None, "Informalidad laboral por región natural, 2015–2025",
    "series_region_natural.png", "Región")

# 5. Comparación 2015 vs 2025 por grupos
print("  construyendo comparacion_2015_vs_2025...")
grupos = [
    ("Nacional", df),
    ("Urbano", df[df["area"] == 1]),  ("Rural", df[df["area"] == 2]),
    ("Hombre", df[df["sexo"] == 1]),  ("Mujer", df[df["sexo"] == 2]),
    ("Costa", df[df["region"] == 1]), ("Sierra", df[df["region"] == 2]),
    ("Selva", df[df["region"] == 3]),
]
labs = [g[0] for g in grupos]
t2015 = [tasa_ponderada(g[1][g[1]["anio"] == 2015]) for g in grupos]
t2025 = [tasa_ponderada(g[1][g[1]["anio"] == 2025]) for g in grupos]
x = np.arange(len(labs)); wdt = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - wdt/2, t2015, wdt, label="2015", color=AZUL)
ax.bar(x + wdt/2, t2025, wdt, label="2025", color=ROJO)
for i, (a, b) in enumerate(zip(t2015, t2025)):
    ax.text(i - wdt/2, a + 0.6, f"{a:.0f}", ha="center", fontsize=8)
    ax.text(i + wdt/2, b + 0.6, f"{b:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right")
ax.set_ylabel("Tasa de informalidad (%)")
ax.set_title("Informalidad laboral: comparación 2015 vs. 2025 por subgrupo", fontsize=12)
ax.legend(frameon=False)
fig.savefig(FIG / "comparacion_2015_vs_2025.png")
plt.close(fig)
print("  → comparacion_2015_vs_2025.png")

# 6. Correlación de indicadores agregados anuales
print("  construyendo correlacion_indicadores...")
agg = df.groupby("anio").apply(lambda d: pd.Series({
    "Tasa informalidad": tasa_ponderada(d),
    "% urbano":   100*np.average((d["area"] == 1), weights=d["fac500a"]),
    "% mujer":    100*np.average((d["sexo"] == 2), weights=d["fac500a"]),
    "% joven 14–24": 100*np.average((d["grupo_edad"] == 1), weights=d["fac500a"]),
    "Edad media": np.average(d["edad"], weights=d["fac500a"]),
    "Nivel educativo": np.average(d["nivel_edu"], weights=d["fac500a"]),
}))
corr = agg.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.values[i,j]) > 0.6 else "black", fontsize=8)
ax.set_title("Correlación de indicadores agregados anuales", fontsize=12)
ax.grid(False)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.savefig(FIG / "correlacion_indicadores.png")
plt.close(fig)
print("  → correlacion_indicadores.png")

# ── Figuras de robustez (desde CSV, sin reentrenar) ──────────────────────────
print("  construyendo figuras de robustez...")
ew = pd.read_csv(TAB / "tabla_robustez_expanding_window.csv")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ew["año_pred"], ew["auc_roc"], marker="o", linewidth=2.4, color=AZUL)
ax.axhline(ew["auc_roc"].mean(), color=ROJO, ls="--", lw=1,
           label=f"AUC medio = {ew['auc_roc'].mean():.4f}")
for _, r in ew.iterrows():
    ax.text(r["año_pred"], r["auc_roc"] + 0.0012, f"{r['auc_roc']:.3f}",
            ha="center", fontsize=8)
ax.set_ylim(0.94, 0.99)
ax.set_xlabel("Año predicho"); ax.set_ylabel("AUC-ROC fuera de muestra")
ax.set_title("Validación cruzada temporal (expanding window) — XGBoost", fontsize=12)
ax.legend(frameon=False); ax.set_xticks(ew["año_pred"])
fig.savefig(FIG / "figura_robustez_expanding_window.png")
plt.close(fig)
print("  → figura_robustez_expanding_window.png")

bs = pd.read_csv(TAB / "tabla_robustez_shap_bootstrap.csv").head(12).iloc[::-1]
pretty = (bs["feature"].str.replace("catocup_", "Cat. ocup. ", regex=False)
          .str.replace("sector_", "Sector ", regex=False)
          .str.replace("_", " ", regex=False))
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(pretty, bs["rank_medio"], xerr=bs["rank_std"], color=VERDE,
        alpha=0.85, capsize=3)
ax.set_xlabel("Rango medio de importancia (bootstrap, 200 remuestras)")
ax.set_title("Estabilidad del ranking SHAP (menor = más importante)", fontsize=12)
ax.invert_xaxis()
fig.savefig(FIG / "figura_robustez_shap_bootstrap.png")
plt.close(fig)
print("  → figura_robustez_shap_bootstrap.png")

print("\n✅ PASO 16 COMPLETADO — 8 figuras en outputs/figures/")
