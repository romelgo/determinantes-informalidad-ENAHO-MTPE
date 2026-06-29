"""
============================================================
PASO 12: Intervalos de confianza por bootstrap — Tabla 2
============================================================

Responde a la Obs. 6 del revisor (Frontiers): las métricas puntuales
(AUC-ROC, F1, Exactitud) se reportan sin incertidumbre. Aquí se cuantifica
el IC 95 % por bootstrap de percentiles (1000 réplicas) sobre el conjunto de
prueba temporal (2022-2025), reutilizando las predicciones ya guardadas en
`models/` (no se reentrena ningún modelo).

Métricas (consistentes con 08_comparacion_modelos.py):
- AUC-ROC : roc_auc_score (independiente del umbral).
- F1      : f1_score con umbral 0.5.
- Exactitud: en el umbral de operación que la maximiza (fijado en la muestra
            completa y mantenido constante en cada réplica bootstrap).

Probit: sus predicciones de test no estaban guardadas; se regeneran de forma
determinista replicando exactamente 05_lasso.py (semilla 42, mismas columnas
exógenas y misma máscara de varianza).

Salida: outputs/tables/tabla2_ic.csv
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from config import PROCESSED, MODELS, TAB

SEED   = 42
N_BOOT = 1000

def load_p(name):
    with open(PROCESSED / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def load_m(name):
    with open(MODELS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

print("=" * 60)
print("PASO 12: IC 95% por bootstrap de las métricas (Tabla 2)")
print("=" * 60)

# ── Datos de prueba ───────────────────────────────────────────────────────────
X_test        = load_p("X_test")
y_test        = load_p("y_test")
feature_names = load_p("feature_names")
y_true        = y_test.values.astype(int)
n = len(y_true)
print(f"  Conjunto de prueba: N={n:,}  tasa informal={y_true.mean():.3f}")

# ── Predicciones guardadas de los modelos enriquecidos ────────────────────────
res_lasso = load_m("resultados_lasso")
res_rf    = load_m("resultados_rf")
res_xgb   = load_m("resultados_xgb")

preds = {
    "LASSO Logístico": res_lasso["y_pred_lasso"],
    "Random Forest":   res_rf["y_pred_rf"],
    "XGBoost":         res_xgb["y_pred_xgb"],
}

# ── Probit: regenerar predicciones de test de forma determinista ──────────────
# Réplica exacta de 05_lasso.py §5.1
print("  Regenerando predicciones de test del Probit (semilla 42)...")
import statsmodels.api as sm
X_train = load_p("X_train")
y_train = load_p("y_train")

_excluir = ("catocup", "tam_emp", "mype", "sector", "horas", "log_ing")
cols_exo = [f for f in feature_names if not f.startswith(_excluir)]

np.random.seed(SEED)
idx_sample = np.random.choice(len(X_train), size=min(50000, len(X_train)), replace=False)
X_probit_arr = X_train.iloc[idx_sample][cols_exo].fillna(0).values
keep_mask = X_probit_arr.var(axis=0) > 0

res_probit = load_m("modelo_probit")          # statsmodels result ya ajustado
X_test_probit = sm.add_constant(X_test[cols_exo].fillna(0).values[:, keep_mask])
preds["Probit (exógeno)"] = np.asarray(res_probit.predict(X_test_probit))

# ── Umbral que maximiza la exactitud (fijado en la muestra completa) ──────────
def best_acc_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    accs = [accuracy_score(y, (p >= t).astype(int)) for t in grid]
    j = int(np.argmax(accs))
    return float(grid[j])

thr = {m: best_acc_threshold(y_true, p) for m, p in preds.items()}

# ── Bootstrap de percentiles ──────────────────────────────────────────────────
def metrics(y, p, t):
    yhat = (p >= 0.5).astype(int)            # F1 a umbral 0.5 (consistente con 08)
    yhat_acc = (p >= t).astype(int)          # exactitud al umbral óptimo
    return (roc_auc_score(y, p),
            f1_score(y, yhat),
            accuracy_score(y, yhat_acc))

orden = ["Probit (exógeno)", "LASSO Logístico", "Random Forest", "XGBoost"]
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

rows = []
for m in orden:
    p = np.asarray(preds[m])
    auc0, f10, acc0 = metrics(y_true, p, thr[m])
    bs = np.array([metrics(y_true[ix], p[ix], thr[m]) for ix in boot_idx])
    lo = np.percentile(bs, 2.5, axis=0)
    hi = np.percentile(bs, 97.5, axis=0)
    rows.append({
        "Modelo": m,
        "AUC": round(auc0, 4), "AUC_lo": round(lo[0], 4), "AUC_hi": round(hi[0], 4),
        "F1":  round(f10, 4),  "F1_lo":  round(lo[1], 4), "F1_hi":  round(hi[1], 4),
        "Exactitud": round(acc0, 4), "Acc_lo": round(lo[2], 4), "Acc_hi": round(hi[2], 4),
        "umbral_acc": round(thr[m], 4),
    })
    print(f"  {m:18s} AUC {auc0:.4f} [{lo[0]:.4f}, {hi[0]:.4f}] | "
          f"F1 {f10:.4f} [{lo[1]:.4f}, {hi[1]:.4f}] | "
          f"Acc {acc0:.4f} [{lo[2]:.4f}, {hi[2]:.4f}]")

tabla = pd.DataFrame(rows)
out = TAB / "tabla2_ic.csv"
tabla.to_csv(out, index=False)
print(f"\n✅ PASO 12 COMPLETADO  → {out}")
