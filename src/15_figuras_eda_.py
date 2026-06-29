"""
============================================================
PASO 15: Figuras EDA 
============================================================
tres figuras descriptivas que el paper referencia como floats:

  outputs/figures/series_temporales_nacionales.png   — informalidad ponderada por año
  outputs/figures/heatmap_departamentos.png          — informalidad por departamento × año
  outputs/figures/figura2_composicion_planilla.png   — composición del empleo formal (Planilla)

Todas se calculan ponderadas por el factor de expansión fac500a y se guardan a 300 dpi.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import INTERIM, FIG

plt.rcParams.update({"figure.dpi": 300, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})
COVID = (2020, 2021)

def shade_covid(ax):
    ax.axvspan(COVID[0] - 0.5, COVID[1] + 0.5, color="red", alpha=0.08, zorder=0)

def tasa(df, keys):
    """Tasa de informalidad ponderada (%) agregada por keys."""
    g = (df.groupby(keys, observed=True)
           .apply(lambda x: 100 * np.average(x["Y"], weights=x["fac500a"])))
    return g.rename("tasa").reset_index()

print("=" * 60)
print("PASO 15: Figuras EDA para el paper")
print("=" * 60)

# ── Cargar ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INTERIM / "dataset_integrado.csv",
                 usecols=["anio", "Y", "fac500a", "area_label", "dep_label"])
plan = pd.read_csv(INTERIM / "planilla_limpia.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Figura 1 — Series temporales nacionales (nacional + urbano/rural)
# ══════════════════════════════════════════════════════════════════════════════
nac = tasa(df, ["anio"])
area = tasa(df, ["anio", "area_label"]).pivot(index="anio", columns="area_label", values="tasa")

fig, ax = plt.subplots(figsize=(9, 5))
shade_covid(ax)
ax.plot(nac["anio"], nac["tasa"], "o-", color="#c0392b", lw=2.6, label="Nacional", zorder=3)
ax.plot(area.index, area["Urbano"], "s--", color="#2980b9", lw=1.8, label="Urbano")
ax.plot(area.index, area["Rural"], "^--", color="#7f8c8d", lw=1.8, label="Rural")
ax.set_xlabel("Año"); ax.set_ylabel("Tasa de informalidad (%)")
ax.set_xticks(range(2015, 2026)); ax.set_ylim(55, 100)
ax.set_title("Tasa de informalidad laboral en Perú, 2015–2025 (ponderada con fac500a)")
ax.annotate("COVID-19", xy=(2020.5, 96), ha="center", fontsize=8, color="#c0392b")
ax.legend(loc="center left", framealpha=0.9)
fig.savefig(FIG / "series_temporales_nacionales.png")
plt.close(fig)
print(f"  → {FIG/'series_temporales_nacionales.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Figura 2 — Heatmap departamento × año
# ══════════════════════════════════════════════════════════════════════════════
dep = tasa(df, ["dep_label", "anio"]).pivot(index="dep_label", columns="anio", values="tasa")
dep = dep.loc[dep.mean(axis=1).sort_values(ascending=False).index]   # ordenar por promedio

fig, ax = plt.subplots(figsize=(10, 8))
ax.grid(False)
im = ax.imshow(dep.values, aspect="auto", cmap="YlOrRd", vmin=50, vmax=100)
ax.set_xticks(range(dep.shape[1])); ax.set_xticklabels(dep.columns, rotation=45, ha="right")
ax.set_yticks(range(dep.shape[0])); ax.set_yticklabels(dep.index, fontsize=8)
for i in range(dep.shape[0]):
    for j in range(dep.shape[1]):
        v = dep.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="black" if v < 82 else "white")
fig.colorbar(im, ax=ax, label="Tasa de informalidad (%)", fraction=0.025, pad=0.02)
ax.set_title("Informalidad laboral por departamento y año (ponderada)")
fig.savefig(FIG / "heatmap_departamentos.png")
plt.close(fig)
print(f"  → {FIG/'heatmap_departamentos.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Figura 3 — Composición del empleo formal (Planilla) por sector
# ══════════════════════════════════════════════════════════════════════════════
sec_cols = ["pct_agropecuario_y_pesca", "pct_manufactura", "pct_comercio",
            "pct_construccion", "pct_mineria_y_canteras", "pct_servicios"]
labels = ["Agro/pesca", "Manufactura", "Comercio", "Construcción", "Minería", "Servicios"]
S = plan.set_index("AÑO")[sec_cols] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
shade_covid(ax1)
ax1.stackplot(S.index, *[S[c] for c in sec_cols], labels=labels, alpha=0.88)
ax1.set_title("Composición del empleo formal por sector (Planilla)")
ax1.set_xlabel("Año"); ax1.set_ylabel("% del empleo formal")
ax1.set_xticks(range(2015, 2026)); ax1.set_ylim(0, 100)
ax1.legend(loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.13))

shade_covid(ax2)
ax2.plot(plan["AÑO"], 100 * plan["pct_mype_formal"], "o-", color="#16a085", lw=2)
ax2.set_title("Participación MYPE en el empleo formal")
ax2.set_xlabel("Año"); ax2.set_ylabel("% MYPE")
ax2.set_xticks(range(2015, 2026, 2))
fig.savefig(FIG / "figura2_composicion_planilla.png")
plt.close(fig)
print(f"  → {FIG/'figura2_composicion_planilla.png'}")

print("\n✅ PASO 15 COMPLETADO — 3 figuras EDA en outputs/figures/")
