"""
============================================================
PASO 8: Tabla comparativa de modelos — Tabla 2 del paper
============================================================


Por qué cada métrica:
- AUC-ROC: mide la capacidad discriminante del modelo. Es preferida para datos
  desbalanceados (más informales que formales) porque no depende del umbral.
  AUC=0.5 → equivale a aleatoriedad; AUC=1.0 → clasificación perfecta.
- F1-score: media armónica de precisión y recall. Útil cuando el costo de
  falsos negativos (clasificar formal a un informal) es importante.
- McFadden R²: (1 - LL_modelo/LL_nulo). Comparable con R² de regresión.
  Valores 0.2–0.4 se consideran excelentes en econometría.
- Curvas de calibración: verifican si las probabilidades predichas son confiables
  (probabilidad predicha de 0.7 ≈ 70% de informales reales). Crítico para
  policy: no solo clasificar, sino estimar probabilidades correctas.

Outputs:
- tablas/tabla2_comparacion_modelos.csv  (+.tex)
- figuras/figura_roc_comparacion.pdf
- figuras/figura_calibracion.pdf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score, RocCurveDisplay,
                             roc_curve)
from sklearn.calibration import CalibrationDisplay, calibration_curve
import warnings
warnings.filterwarnings("ignore")

from config import ROOT, PROCESSED, MODELS, FIG, TAB
DATOS   = PROCESSED
MODELOS = MODELS
TABLAS  = TAB
FIGS    = FIG

print("=" * 60)
print("PASO 8: Comparación de Modelos — Tabla 2")
print("=" * 60)

def load(name):
    with open(DATOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def load_m(name):
    with open(MODELOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

X_test        = load("X_test")
X_test_s      = load("X_test_scaled")
y_test        = load("y_test")
feature_names = load("feature_names")

X_test_v  = X_test.fillna(0).values
X_test_sv = X_test_s.fillna(0).values
y_test_v  = y_test.values

# ── Cargar modelos y predicciones guardadas ───────────────────────────────────
resultados_lasso = load_m("resultados_lasso")
resultados_rf    = load_m("resultados_rf")
resultados_xgb   = load_m("resultados_xgb")

y_pred_lasso = resultados_lasso["y_pred_lasso"]
y_pred_rf    = resultados_rf["y_pred_rf"]
y_pred_xgb   = resultados_xgb["y_pred_xgb"]

# ── Re-calcular métricas estandarizadas ───────────────────────────────────────
def mcfadden(y_true, y_pred_prob, base_rate=None):
    """McFadden R² aproximado."""
    if base_rate is None:
        base_rate = y_true.mean()
    eps = 1e-15
    ll_null  = len(y_true) * (base_rate * np.log(base_rate + eps) + 
                              (1-base_rate) * np.log(1-base_rate + eps))
    ll_model = (y_true * np.log(np.clip(y_pred_prob, eps, 1-eps)) + 
                (1-y_true) * np.log(np.clip(1-y_pred_prob, eps, 1-eps))).sum()
    return 1 - ll_model / ll_null if ll_null != 0 else np.nan

base_rate = y_test_v.mean()
modelos_info = [
    {
        "Modelo":         "Probit Baseline",
        "AUC-ROC":        resultados_lasso["probit"].get("auc_roc"),
        "F1":             resultados_lasso["probit"].get("f1"),
        "McFadden_R2":    resultados_lasso["probit"].get("mcfadden_r2"),
        "N_variables":    resultados_lasso["probit"].get("n_variables"),
        "y_pred":         None,
    },
    {
        "Modelo":         "LASSO Logístico",
        "AUC-ROC":        round(roc_auc_score(y_test_v, y_pred_lasso), 4),
        "F1":             round(f1_score(y_test_v, (y_pred_lasso>0.5).astype(int)), 4),
        "McFadden_R2":    round(mcfadden(y_test_v, y_pred_lasso, base_rate), 4),
        "N_variables":    resultados_lasso["lasso"]["n_variables"],
        "y_pred":         y_pred_lasso,
    },
    {
        "Modelo":         "Random Forest",
        "AUC-ROC":        round(roc_auc_score(y_test_v, y_pred_rf), 4),
        "F1":             round(f1_score(y_test_v, (y_pred_rf>0.5).astype(int)), 4),
        "McFadden_R2":    round(mcfadden(y_test_v, y_pred_rf, base_rate), 4),
        "N_variables":    resultados_rf["rf"]["n_variables"],
        "y_pred":         y_pred_rf,
    },
    {
        "Modelo":         "XGBoost",
        "AUC-ROC":        round(roc_auc_score(y_test_v, y_pred_xgb), 4),
        "F1":             round(f1_score(y_test_v, (y_pred_xgb>0.5).astype(int)), 4),
        "McFadden_R2":    round(mcfadden(y_test_v, y_pred_xgb, base_rate), 4),
        "N_variables":    resultados_xgb["xgb"]["n_variables"],
        "y_pred":         y_pred_xgb,
    },
]

tabla2 = pd.DataFrame([{k: v for k, v in m.items() if k != "y_pred"} for m in modelos_info])
print("\n  TABLA 2 — Comparación de Modelos:")
print(tabla2.to_string(index=False))
tabla2.to_csv(TABLAS / "tabla2_comparacion_modelos.csv", index=False)

# LaTeX
latex = []
latex.append(r"\begin{table}[htbp]")
latex.append(r"\centering")
latex.append(r"\caption{Comparación de modelos predictivos de informalidad laboral}")
latex.append(r"\label{tab:modelos}")
latex.append(r"\begin{tabular}{lcccc}")
latex.append(r"\hline\hline")
latex.append(r"Modelo & AUC-ROC & F1 & McFadden $R^2$ & N variables \\")
latex.append(r"\hline")
for _, row in tabla2.iterrows():
    auc = f"{row['AUC-ROC']:.4f}" if pd.notna(row['AUC-ROC']) else "--"
    f1  = f"{row['F1']:.4f}" if pd.notna(row['F1']) else "--"
    mf  = f"{row['McFadden_R2']:.4f}" if pd.notna(row['McFadden_R2']) else "--"
    nv  = str(int(row['N_variables'])) if pd.notna(row['N_variables']) else "--"
    latex.append(f"{row['Modelo']} & {auc} & {f1} & {mf} & {nv} \\\\")
latex.append(r"\hline\hline")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")
with open(TABLAS / "tabla2_comparacion_modelos.tex", "w") as f:
    f.write("\n".join(latex))

# ══════════════════════════════════════════════════════════════════════════════
# 8.1 — Figura: ROC Curves superpuestas
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8.1] Generando figura ROC curves superpuestas...")
colores = {"LASSO Logístico": "#2563EB", "Random Forest": "#16A34A", "XGBoost": "#DC2626"}

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0,1],[0,1],"--", color="gray", linewidth=1, label="Aleatoriedad (AUC=0.50)")
for m in modelos_info:
    if m["y_pred"] is None:
        continue
    fpr, tpr, _ = roc_curve(y_test_v, m["y_pred"])
    auc = roc_auc_score(y_test_v, m["y_pred"])
    ax.plot(fpr, tpr, linewidth=2, color=colores.get(m["Modelo"],"black"),
            label=f"{m['Modelo']} (AUC = {auc:.4f})")
ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
ax.set_title("Figura: Curvas ROC — Comparación de Modelos\nPredicción de Informalidad Laboral en Perú", fontsize=10)
ax.legend(frameon=False, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(FIGS / "figura_roc_comparacion.pdf")
plt.savefig(FIGS / "figura_roc_comparacion.png", dpi=300)
plt.close()
print(f"  ✅ Figura ROC guardada")

# ══════════════════════════════════════════════════════════════════════════════
# 8.2 — Figura: Calibration Curves
# ══════════════════════════════════════════════════════════════════════════════
print("[8.2] Generando figura Calibration curves...")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0,1],[0,1],"--", color="gray", linewidth=1, label="Calibración perfecta")
for m in modelos_info:
    if m["y_pred"] is None:
        continue
    prob_true, prob_pred = calibration_curve(y_test_v, m["y_pred"], n_bins=10)
    ax.plot(prob_pred, prob_true, marker="o", markersize=4, linewidth=1.8,
            color=colores.get(m["Modelo"],"black"), label=m["Modelo"])
ax.set_xlabel("Probabilidad predicha")
ax.set_ylabel("Fracción de positivos reales")
ax.set_title("Figura B1. Curvas de Calibración — Comparación de Modelos\n"
             "(Apéndice B)", fontsize=10)
ax.legend(frameon=False)
ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(FIGS / "figura_calibracion.pdf")
plt.savefig(FIGS / "figura_calibracion.png", dpi=300)
plt.close()
print(f"  ✅ Figura Calibración guardada")

print(f"\n✅ PASO 8 COMPLETADO")
print(f"   → tablas/tabla2_comparacion_modelos.csv")
print(f"   → figuras/figura_roc_comparacion.pdf")
print(f"   → figuras/figura_calibracion.pdf")
