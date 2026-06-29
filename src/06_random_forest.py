"""
============================================================
PASO 6: Random Forest — captura interacciones no lineales
============================================================

Por qué Random Forest:
- Bootstrap Aggregating (bagging): entrena B árboles en submuestras bootstrap
  y promedia predicciones → reduce varianza sin aumentar sesgo.
- Captura interacciones no lineales que LASSO (modelo aditivo) no detecta.
  Ejemplo: el efecto de ser joven puede ser distinto según el sector.
- Out-of-Bag (OOB) error: cada árbol se evalúa en el ~37% de datos excluidos
  de su muestra bootstrap → estimación gratuita del error sin usar test set.
- Importancia por permutación: se baraja aleatoriamente una variable y se mide
  la caída en AUC-ROC → qué tanto depende el modelo de esa variable.
- Robustez a multicolinealidad: en cada split solo se considera un subconjunto
  aleatorio de features (max_features='sqrt'), diluyendo la colinealidad.

Nota: NO se escalan las features (RF es invariante a escala).

Outputs:
- modelos/modelo_rf.pkl
- tablas/importancia_rf.csv
- figuras/figura3a_importancia_rf.pdf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

from config import ROOT, PROCESSED, MODELS, FIG, TAB
DATOS   = PROCESSED
MODELOS = MODELS
TABLAS  = TAB
FIGS    = FIG
MODELOS.mkdir(exist_ok=True)

print("=" * 60)
print("PASO 6: Random Forest")
print("=" * 60)

def load(name):
    with open(DATOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

# Usar datos SIN escalar (RF es invariante a escala)
X_train       = load("X_train")
X_test        = load("X_test")
y_train       = load("y_train")
y_test        = load("y_test")
w_train       = load("w_train")
feature_names = load("feature_names")

X_train_v = X_train.fillna(0).values
X_test_v  = X_test.fillna(0).values
y_train_v = y_train.values
y_test_v  = y_test.values
w_train_v = w_train.values

print(f"  Train: {X_train_v.shape}, Test: {X_test_v.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.1 — Grid Search con validación cruzada 5-fold
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6.1] Grid Search Random Forest (cv=5, scoring='roc_auc')...")
print("  (Puede tomar varios minutos con 400K+ obs — usando submuestra para grid)")

# Grid definido en el prompt
param_grid = {
    "n_estimators":    [300, 500],
    "max_depth":       [8, 12, None],
    "min_samples_leaf":[50, 100],
    "max_features":    ["sqrt", 0.3],
}

# Submuestra estratificada para el grid search (velocidad)
np.random.seed(42)
from sklearn.model_selection import train_test_split
idx_gs = np.random.choice(len(X_train_v), size=min(80000, len(X_train_v)), replace=False)
X_gs = X_train_v[idx_gs]
y_gs = y_train_v[idx_gs]
w_gs = w_train_v[idx_gs]

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=cv5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_gs, y_gs, sample_weight=w_gs)
print(f"\n  Mejores hiperparámetros: {grid_search.best_params_}")
print(f"  Mejor AUC-ROC (CV): {grid_search.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.2 — Entrenar modelo final con mejores hiperparámetros en dataset completo
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6.2] Entrenando modelo final con dataset completo...")
best_params = grid_search.best_params_
rf_final = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
rf_final.fit(X_train_v, y_train_v, sample_weight=w_train_v)
print(f"  OOB AUC aprox (OOB accuracy): {rf_final.oob_score_:.4f}")

# ── Métricas en test ─────────────────────────────────────────────────────────
y_pred_prob  = rf_final.predict_proba(X_test_v)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype(int)
auc_rf       = roc_auc_score(y_test_v, y_pred_prob)
f1_rf        = f1_score(y_test_v, y_pred_class)

print(f"\n  📊 Métricas Random Forest en test:")
print(f"     AUC-ROC:  {auc_rf:.4f}  (esperado: 0.79–0.84)")
print(f"     F1-score: {f1_rf:.4f}  (esperado: 0.75–0.80)")
print(f"\n  {classification_report(y_test_v, y_pred_class, target_names=['Formal','Informal'])}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.3 — Importancia por permutación (top 15 variables)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6.3] Calculando importancia por permutación (puede tomar 2-3 min)...")
# Submuestra para permutation importance (eficiencia)
idx_pi = np.random.choice(len(X_test_v), size=min(20000, len(X_test_v)), replace=False)
perm_imp = permutation_importance(
    rf_final,
    X_test_v[idx_pi],
    y_test_v[idx_pi],
    n_repeats=10,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
)
importancia = pd.DataFrame({
    "feature":     feature_names,
    "importancia": perm_imp.importances_mean,
    "std":         perm_imp.importances_std,
}).sort_values("importancia", ascending=False)
importancia.to_csv(TABLAS / "importancia_rf.csv", index=False)
print(f"\n  Top 15 variables más importantes:")
print(importancia.head(15).to_string(index=False))

# ── Figura 3a: Barplot horizontal de importancia (top 15) ───────────────────
top15 = importancia.head(15).copy()
top15_sorted = top15.sort_values("importancia")

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top15_sorted)))
bars = ax.barh(
    top15_sorted["feature"],
    top15_sorted["importancia"],
    xerr=top15_sorted["std"],
    color=colors,
    edgecolor="white",
    linewidth=0.5,
    error_kw={"elinewidth": 0.8, "ecolor": "gray", "capsize": 3},
)
ax.set_xlabel("Disminución en AUC-ROC al permutar la variable")
ax.set_title("Figura 3a. Importancia de variables por permutación\n"
             "Random Forest — Top 15 predictores de informalidad",
             fontsize=10)
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig(FIGS / "figura3a_importancia_rf.pdf")
plt.savefig(FIGS / "figura3a_importancia_rf.png", dpi=300)
plt.close()
print(f"\n  ✅ Figura 3a guardada")

# ── Guardar modelo ────────────────────────────────────────────────────────────
with open(MODELOS / "modelo_rf.pkl", "wb") as f:
    pickle.dump(rf_final, f)

rf_results = {
    "modelo":      "Random Forest",
    "auc_roc":     round(auc_rf, 4),
    "f1":          round(f1_rf, 4),
    "mcfadden_r2": None,
    "n_variables": len(feature_names),
    "params":      best_params,
}
with open(MODELOS / "resultados_rf.pkl", "wb") as f:
    pickle.dump({"rf": rf_results, "y_pred_rf": y_pred_prob}, f)

print(f"\n✅ PASO 6 COMPLETADO")
print(f"   AUC-ROC RF: {auc_rf:.4f}")
print(f"   → modelos/modelo_rf.pkl")
print(f"   → tablas/importancia_rf.csv")
print(f"   → figuras/figura3a_importancia_rf.pdf")
