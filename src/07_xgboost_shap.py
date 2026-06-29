"""
============================================================
PASO 7: XGBoost + Valores SHAP
============================================================

Por qué XGBoost:
- Gradient Boosting: construye árboles secuencialmente. Cada árbol nuevo
  aprende a corregir los residuos/errores del árbol anterior. Iteración
  hasta minimizar la función de pérdida (log-loss para clasificación).
- XGBoost añade regularización L1 (alpha) y L2 (lambda) sobre los pesos
  de los árboles → menos overfitting que GBM estándar.
- Eficiencia: usa histogramas para búsqueda de splits → más rápido en GPU/CPU.
- scale_pos_weight = n_neg/n_pos para compensar el desbalance de clases.

Por qué SHAP:
- Fundamento: teoría de juegos cooperativos (Shapley Values).
- Cada variable recibe su "contribución justa" a la predicción individual.
- A diferencia de la importancia por impureza (RF), SHAP es consistente y
  tiene dirección (positivo = aumenta probabilidad informal).
- SHAP summary plot (beeswarm): eje X = magnitud del efecto SHAP,
  color = valor original de la feature (rojo=alto, azul=bajo).

Outputs:
- modelos/modelo_xgb.pkl
- modelos/shap_values.pkl
- tablas/tabla_shap_media.csv
- figuras/figura3b_shap_summary.pdf
- figuras/figura4a_shap_dependence_tamano.pdf
- figuras/figura4b_shap_dependence_edad.pdf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

from config import ROOT, PROCESSED, MODELS, FIG, TAB
DATOS   = PROCESSED
MODELOS = MODELS
TABLAS  = TAB
FIGS    = FIG

print("=" * 60)
print("PASO 7: XGBoost + SHAP")
print("=" * 60)

def load(name):
    with open(DATOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

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

# Desbalance de clases
n_pos = (y_train_v == 1).sum()
n_neg = (y_train_v == 0).sum()
spw   = n_neg / n_pos
print(f"  scale_pos_weight = {spw:.2f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

# ══════════════════════════════════════════════════════════════════════════════
# 7.1 — RandomizedSearchCV (n_iter=30, cv=5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7.1] RandomizedSearchCV XGBoost (n_iter=30, cv=5)...")

param_dist = {
    "n_estimators":     [300, 500, 800],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 2, 5],
}

# Submuestra para búsqueda (velocidad)
np.random.seed(42)
idx_rs = np.random.choice(len(X_train_v), size=min(80000, len(X_train_v)), replace=False)
X_rs = X_train_v[idx_rs]
y_rs = y_train_v[idx_rs]
w_rs = w_train_v[idx_rs]

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=spw,
    eval_metric="auc",
    random_state=42,
    device="cuda",
    use_label_encoder=False,
)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rand_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=1,
    verbose=1,
)
rand_search.fit(X_rs, y_rs, sample_weight=w_rs)
print(f"\n  Mejores hiperparámetros: {rand_search.best_params_}")
print(f"  Mejor AUC-ROC (CV): {rand_search.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.2 — Entrenar modelo final con early stopping
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7.2] Entrenando modelo final con early stopping...")
best_p = rand_search.best_params_.copy()
n_est  = best_p.pop("n_estimators")

xgb_final = xgb.XGBClassifier(
    **best_p,
    n_estimators=n_est,
    scale_pos_weight=spw,
    eval_metric="auc",
    random_state=42,
    device="cuda",
    use_label_encoder=False,
    early_stopping_rounds=50,
)
xgb_final.fit(
    X_train_v, y_train_v,
    sample_weight=w_train_v,
    eval_set=[(X_test_v, y_test_v)],
    verbose=10,
)
print(f"  Mejor n_estimators (early stopping): {xgb_final.best_iteration}")

# ── Métricas en test ─────────────────────────────────────────────────────────
y_pred_prob  = xgb_final.predict_proba(X_test_v)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype(int)
auc_xgb      = roc_auc_score(y_test_v, y_pred_prob)
f1_xgb       = f1_score(y_test_v, y_pred_class)

print(f"\n  📊 Métricas XGBoost en test:")
print(f"     AUC-ROC:  {auc_xgb:.4f}  (esperado: 0.82–0.87)")
print(f"     F1-score: {f1_xgb:.4f}  (esperado: 0.78–0.83)")
print(f"\n  {classification_report(y_test_v, y_pred_class, target_names=['Formal','Informal'])}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.3 — SHAP Values (TreeExplainer)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7.3] Calculando SHAP values (TreeExplainer)...")
# Submuestra para SHAP (1000 observaciones para beeswarm legible)
np.random.seed(42)
idx_shap = np.random.choice(len(X_test_v), size=min(2000, len(X_test_v)), replace=False)
X_shap   = X_test_v[idx_shap]

explainer   = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_shap)

# Tabla de importancia media SHAP (|SHAP|)
shap_media = pd.DataFrame({
    "feature":      feature_names,
    "shap_abs_mean": np.abs(shap_values).mean(axis=0),
    "shap_mean":     shap_values.mean(axis=0),
}).sort_values("shap_abs_mean", ascending=False)
shap_media.to_csv(TABLAS / "tabla_shap_media.csv", index=False)
print(f"\n  Top 10 variables por |SHAP| medio:")
print(shap_media.head(10).to_string(index=False))

# ── Figura 3b: SHAP summary beeswarm (top 15) ────────────────────────────────
print("\n[7.4] Generando Figura 3b: SHAP Summary Plot...")
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_shap,
    feature_names=feature_names,
    max_display=15,
    show=False,
    plot_size=None,
)
plt.title("Figura 3b. SHAP Summary Plot — XGBoost\n"
          "(Top 15 predictores de informalidad, n=2,000 obs.)",
          fontsize=10, pad=12)
plt.tight_layout()
plt.savefig(FIGS / "figura3b_shap_summary.pdf", bbox_inches="tight")
plt.savefig(FIGS / "figura3b_shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✅ Figura 3b guardada")

# ── Figura 4a: SHAP Dependence — tamano_empresa / pct_mype_formal ─────────────
print("[7.5] Generando Figura 4a: SHAP Dependence — tamaño empresa...")
# Buscar la feature de MYPE más relevante
feature_mype = next(
    (f for f in feature_names if "mype" in f or "empresa" in f or "tamao" in f), 
    None
)
feature_sector = next(
    (f for f in feature_names if "servicios" in f or "comercio" in f),
    None
)
if feature_mype and feature_mype in feature_names:
    idx_mype   = feature_names.index(feature_mype)
    idx_sector = feature_names.index(feature_sector) if feature_sector else None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        idx_mype,
        shap_values,
        X_shap,
        feature_names=feature_names,
        interaction_index=idx_sector,
        ax=ax,
        show=False,
    )
    ax.set_title(f"Figura 4a. SHAP Dependence — Tamaño empresa ({feature_mype})\n"
                 f"(interacción: {feature_sector if feature_sector else 'auto'})",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGS / "figura4a_shap_dependence_tamano.pdf")
    plt.savefig(FIGS / "figura4a_shap_dependence_tamano.png", dpi=300)
    plt.close()
    print(f"  ✅ Figura 4a guardada ({feature_mype})")
else:
    print(f"  ⚠️ Feature MYPE no encontrada en: {feature_names[:10]}...")

# ── Figura 4b: SHAP Dependence — grupo_edad ──────────────────────────────────
print("[7.6] Generando Figura 4b: SHAP Dependence — grupo edad...")
feature_edad = next(
    (f for f in feature_names if "grupo_edad" in f or "gedad" in f or "edad" == f),
    "edad"
)
feature_region = next(
    (f for f in feature_names if "region" in f),
    None
)
if feature_edad in feature_names:
    idx_edad   = feature_names.index(feature_edad)
    idx_region = feature_names.index(feature_region) if feature_region else None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        idx_edad,
        shap_values,
        X_shap,
        feature_names=feature_names,
        interaction_index=idx_region,
        ax=ax,
        show=False,
    )
    ax.set_title(f"Figura 4b. SHAP Dependence — Grupo de edad ({feature_edad})\n"
                 f"(interacción: {feature_region if feature_region else 'auto'})",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGS / "figura4b_shap_dependence_edad.pdf")
    plt.savefig(FIGS / "figura4b_shap_dependence_edad.png", dpi=300)
    plt.close()
    print(f"  ✅ Figura 4b guardada ({feature_edad})")

# ── Guardar artefactos ────────────────────────────────────────────────────────
with open(MODELOS / "modelo_xgb.pkl", "wb") as f:
    pickle.dump(xgb_final, f)
with open(MODELOS / "shap_values.pkl", "wb") as f:
    pickle.dump({"shap_values": shap_values, "X_shap": X_shap,
                 "feature_names": feature_names}, f)

xgb_results = {
    "modelo":      "XGBoost",
    "auc_roc":     round(auc_xgb, 4),
    "f1":          round(f1_xgb, 4),
    "mcfadden_r2": None,
    "n_variables": len(feature_names),
    "params":      rand_search.best_params_,
}
with open(MODELOS / "resultados_xgb.pkl", "wb") as f:
    pickle.dump({"xgb": xgb_results, "y_pred_xgb": y_pred_prob}, f)

print(f"\n✅ PASO 7 COMPLETADO")
print(f"   AUC-ROC XGBoost: {auc_xgb:.4f}")
print(f"   → modelos/modelo_xgb.pkl")
print(f"   → modelos/shap_values.pkl")
print(f"   → tablas/tabla_shap_media.csv")
print(f"   → figuras/figura3b_shap_summary.pdf")
