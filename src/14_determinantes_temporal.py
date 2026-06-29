"""
============================================================
PASO 14: Determinantes de la informalidad y su evolución temporal
============================================================
Responde directamente la pregunta del proyecto:
  «¿Qué factores tienen MÁS PESO en la informalidad y cómo CAMBIAN
   a lo largo de 2015-2025, incluido el shock COVID?»

Dos entregables:
  14.1  Ranking CONSOLIDADO de determinantes — combina la evidencia de los
        tres modelos ya entrenados (Random Forest, LASSO y XGBoost+SHAP) en
        una sola tabla y figura, a nivel de variable y de FAMILIA de factor.
  14.2  SHAP por PERIODO — magnitud media de contribución de cada factor en
        tres ventanas: pre-COVID (2015-2019), COVID (2020-2021) y post (2022-2025).
        Muestra qué determinantes ganan o pierden peso en el tiempo.

CAVEAT (documentado en el paper): categoría ocupacional, tamaño de empresa y
sector se relacionan parcialmente por construcción con la definición de
informalidad → lectura PREDICTIVA/operativa, no causal. La lectura causal del
shock COVID se hace por separado con Difference-in-Differences (src/10_did_covid.py).

Outputs:
  outputs/tables/determinantes_ranking.csv          (por variable)
  outputs/tables/determinantes_ranking_familias.csv (por familia de factor)
  outputs/tables/determinantes_por_periodo.csv      (|SHAP| medio × periodo)
  outputs/figures/determinantes_ranking.png/.pdf
  outputs/figures/shap_por_periodo.png/.pdf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import PROCESSED, MODELS, TAB, FIG

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ──────────────────────────────────────────────────────────────────────────────
# Mapeo de variable (one-hot) → FAMILIA de factor legible
# ──────────────────────────────────────────────────────────────────────────────
def familia(feat: str) -> str:
    f = feat.lower()
    if f.startswith("catocup"):        return "Categoría ocupacional"
    if f == "mype" or f.startswith("tam_emp"): return "Tamaño de empresa (MYPE)"
    if f.startswith("sector"):         return "Sector económico"
    if f == "horas":                   return "Horas trabajadas"
    if f == "log_ing":                 return "Ingreso laboral"
    if f == "edad" or f.startswith("grupo_edad"): return "Edad"
    if f == "sexo":                    return "Sexo"
    if f == "area":                    return "Área (urbano/rural)"
    if f.startswith("region"):         return "Región natural"
    if f.startswith("departamento"):   return "Departamento"
    if f.startswith("nivel_edu"):      return "Nivel educativo"
    if f == "anio_covid":              return "Periodo COVID"
    if f.startswith("anio"):           return "Año"
    if f.startswith("pct_"):           return "Contexto sector formal (Planilla)"
    return "Otros"

# ══════════════════════════════════════════════════════════════════════════════
# 14.1 — RANKING CONSOLIDADO (RF + LASSO + XGBoost/SHAP)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("PASO 14.1: Ranking consolidado de determinantes")
print("=" * 64)

rf   = pd.read_csv(TAB / "importancia_rf.csv")[["feature", "importancia"]]
las  = pd.read_csv(TAB / "tabla_coeficientes_lasso.csv")
shp  = pd.read_csv(TAB / "tabla_shap_media.csv")[["feature", "shap_abs_mean"]]

las["lasso_abs"] = las["coeficiente"].abs()

df = (rf.merge(las[["feature", "coeficiente", "lasso_abs"]], on="feature", how="outer")
        .merge(shp, on="feature", how="outer"))

# Normalización min-max (0-1) por método → score consolidado comparable
def norm(s):
    s = s.fillna(0.0)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0

df["rf_norm"]    = norm(df["importancia"])
df["lasso_norm"] = norm(df["lasso_abs"])
df["shap_norm"]  = norm(df["shap_abs_mean"])
df["score_consolidado"] = df[["rf_norm", "lasso_norm", "shap_norm"]].mean(axis=1)
df["familia"] = df["feature"].map(familia)
df = df.sort_values("score_consolidado", ascending=False).reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)

cols = ["rank", "feature", "familia", "importancia", "coeficiente",
        "shap_abs_mean", "rf_norm", "lasso_norm", "shap_norm", "score_consolidado"]
df[cols].to_csv(TAB / "determinantes_ranking.csv", index=False)
print(f"  → {TAB/'determinantes_ranking.csv'}  ({len(df)} variables)")
print("\n  TOP 12 variables:")
print(df[["rank", "feature", "familia", "score_consolidado"]].head(12).to_string(index=False))

# Agregado por FAMILIA (suma de scores normalizados de sus variables)
fam = (df.groupby("familia")
         .agg(score_total=("score_consolidado", "sum"),
              shap_total=("shap_abs_mean", "sum"),
              n_vars=("feature", "count"))
         .sort_values("score_total", ascending=False)
         .reset_index())
fam.to_csv(TAB / "determinantes_ranking_familias.csv", index=False)
print("\n  RANKING POR FAMILIA DE FACTOR:")
print(fam.to_string(index=False))

# Figura: top-15 variables por score consolidado
top = df.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top["feature"], top["score_consolidado"], color="#2c7fb8")
ax.set_xlabel("Score consolidado (RF + LASSO + SHAP, normalizado 0–1)")
ax.set_title("Determinantes con mayor peso en la informalidad laboral\n(consenso de 3 modelos, 2015–2025)")
fig.tight_layout()
fig.savefig(FIG / "determinantes_ranking.png", bbox_inches="tight")
fig.savefig(FIG / "determinantes_ranking.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  → {FIG/'determinantes_ranking.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# 14.2 — SHAP POR PERIODO (pre-COVID / COVID / post)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("PASO 14.2: SHAP por periodo (evolución temporal del peso)")
print("=" * 64)

def load(name):
    with open(PROCESSED / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)

X_train, X_test = load("X_train"), load("X_test")
anio = pd.concat([load("anio_train"), load("anio_test")], ignore_index=True).astype(int)
w    = pd.concat([load("w_train"),    load("w_test")],    ignore_index=True).astype(float)
X    = pd.concat([X_train, X_test], ignore_index=True)
feature_names = load("feature_names")
X = X[feature_names]
print(f"  Matriz completa: {X.shape}, años {anio.min()}–{anio.max()}")

# Modelo XGBoost ya entrenado (src/07); SHAP = atribución comparable entre periodos
with open(MODELS / "modelo_xgb.pkl", "rb") as fh:
    modelo_xgb = pickle.load(fh)

PERIODOS = {
    "Pre-COVID (2015-19)": (2015, 2019),
    "COVID (2020-21)":     (2020, 2021),
    "Post (2022-25)":      (2022, 2025),
}
RNG = np.random.default_rng(42)
N_MAX = 40_000   # muestra por periodo para acelerar el cálculo de SHAP

explainer = shap.TreeExplainer(modelo_xgb)
rows = {}
for nombre, (a0, a1) in PERIODOS.items():
    idx = np.where((anio >= a0) & (anio <= a1))[0]
    if len(idx) > N_MAX:
        idx = RNG.choice(idx, N_MAX, replace=False)
    Xs = X.iloc[idx]
    ws = w.iloc[idx].to_numpy()
    sv = explainer.shap_values(Xs)
    # |SHAP| medio ponderado por factor de expansión (representatividad poblacional)
    abs_w = np.average(np.abs(sv), axis=0, weights=ws)
    rows[nombre] = pd.Series(abs_w, index=feature_names)
    print(f"  {nombre:22s}: n={len(idx):,}  (informalidad media en periodo)")

shap_per = pd.DataFrame(rows)
shap_per["familia"] = [familia(f) for f in shap_per.index]

# Agregar a familia (suma de |SHAP| de sus variables) para lectura clara
fam_per = shap_per.groupby("familia")[list(PERIODOS)].sum()
fam_per["promedio"] = fam_per.mean(axis=1)
fam_per = fam_per.sort_values("promedio", ascending=False)
fam_per.to_csv(TAB / "determinantes_por_periodo.csv")
print(f"\n  → {TAB/'determinantes_por_periodo.csv'}")
print("\n  |SHAP| medio por FAMILIA y periodo:")
print(fam_per.round(4).to_string())

# Heatmap familias × periodo (top por promedio)
H = fam_per.drop(columns="promedio").head(12)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(H.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=20, ha="right")
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center",
                color="black" if H.values[i, j] < H.values.max() * 0.6 else "white", fontsize=8)
fig.colorbar(im, ax=ax, label="|SHAP| medio (peso del factor)")
ax.set_title("Evolución del peso de los determinantes de informalidad\n|SHAP| medio por familia de factor y periodo")
fig.tight_layout()
fig.savefig(FIG / "shap_por_periodo.png", bbox_inches="tight")
fig.savefig(FIG / "shap_por_periodo.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  → {FIG/'shap_por_periodo.png'}")

print("\n✅ PASO 14 COMPLETADO")
