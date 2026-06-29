"""
============================================================
PASO 5: LASSO Logístico — modelo base interpretable
============================================================

Por qué LASSO logístico:
- La penalización L1 lleva coeficientes exactamente a cero → selección automática
  de variables relevantes (a diferencia de Ridge/L2 que solo los encoge).
- Produce un modelo parsimonioso e interpretable para el paper.
- Validación cruzada de 5 pliegues elige el hiperparámetro λ óptimo
  (representado como C = 1/λ en scikit-learn).
- Los coeficientes tienen una interpretación similar a efectos marginales en probit.

Nota: Se usa X_train_scaled porque LASSO es SENSIBLE a la escala
      (variables con mayor varianza dominarían la penalización sin escalar).

Por qué sklearn en lugar de PyTorch personalizado:
- LogisticRegressionCV con solver='saga' es altamente optimizado, soporta L1,
  y paraleliza los 5 folds con n_jobs=-1.  Mucho más rápido y estable.

Outputs:
- modelos/modelo_lasso.pkl
- tablas/tabla_coeficientes_lasso.csv
- figuras/figura_lasso_regularization_path.pdf
"""

import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── Helper de tiempo ─────────────────────────────────────────────────────────
def elapsed(t0):
    s = time.time() - t0
    return f"{int(s//60)}m{int(s%60):02d}s"

from config import ROOT, PROCESSED, MODELS, FIG, TAB
DATOS   = PROCESSED
MODELOS = MODELS
TABLAS  = TAB
FIGS    = FIG
MODELOS.mkdir(exist_ok=True)
TABLAS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

print("=" * 60)
print("PASO 5: LASSO Logístico")
print("=" * 60)

# ── Cargar artefactos del Paso 4 ─────────────────────────────────────────────
def load(name):
    with open(DATOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

X_train_s     = load("X_train_scaled")
X_test_s      = load("X_test_scaled")
X_train       = load("X_train")
X_test        = load("X_test")
y_train       = load("y_train")
y_test        = load("y_test")
w_train       = load("w_train")
w_test        = load("w_test")
feature_names = load("feature_names")

print(f"  Train: {X_train_s.shape}, Test: {X_test_s.shape}")
print(f"  Features: {len(feature_names)}")

# ── Verificación rápida de varianza (debe ser 0 tras el filtro de 04_features) ─
var_check = X_train_s.var()
zero_var = var_check[var_check == 0]
if len(zero_var) > 0:
    print(f"  ⚠️  Aún hay {len(zero_var)} columnas con varianza 0 en X_train_scaled:")
    print(f"     {list(zero_var.index)}")
    print("     → Eliminando para evitar Singular Matrix...")
    cols_ok = list(var_check[var_check > 0].index)
    X_train_s = X_train_s[cols_ok]
    X_test_s  = X_test_s[cols_ok]
    X_train   = X_train[cols_ok]
    X_test    = X_test[cols_ok]
    feature_names = cols_ok
    print(f"     Features definitivas: {len(feature_names)}")
else:
    print("  ✅ Sin columnas de varianza cero.")

# ══════════════════════════════════════════════════════════════════════════════
# 5.1 — Probit baseline (para comparación con literatura)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5.1] Estimando Probit baseline (statsmodels)...")
# Por qué muestra de 50K: el Probit de statsmodels usa Newton-Raphson que escala
# O(n * p²) — con 1M filas y 58 features sería inmanejable en memoria.
np.random.seed(42)
# Baseline tipo-literatura: el Probit se estima SOLO sobre variables exógenas
# (demografía, geografía, educación, contexto Planilla), EXCLUYENDO los predictores
# estructurales del puesto (categoría ocupacional, tamaño de empresa, sector, horas,
# ingreso). Razones: (i) representa el techo de los modelos paramétricos clásicos de
# la literatura (Yamada 1996; Gasparini & Tornarolli 2009), permitiendo cuantificar
# cuánto aporta enriquecer con variables del puesto; (ii) las variables estructurales
# producen separación cuasi-perfecta que impide la convergencia de Newton-Raphson.
_excluir_probit = ("catocup", "tam_emp", "mype", "sector", "horas", "log_ing")
cols_exo = [f for f in feature_names if not f.startswith(_excluir_probit)]
idx_cols_exo = [feature_names.index(f) for f in cols_exo]
print(f"  Probit baseline sobre {len(cols_exo)} variables EXÓGENAS "
      f"(excluye predictores estructurales del puesto)")

idx_sample = np.random.choice(len(X_train), size=min(50000, len(X_train)), replace=False)
X_probit_arr = X_train.iloc[idx_sample][cols_exo].fillna(0).values

# Eliminar columnas con varianza cero en la muestra (pueden aparecer si la muestra
# aleatoria no captura alguna categoría poco frecuente)
var_sample = X_probit_arr.var(axis=0)
keep_mask  = var_sample > 0
X_probit   = sm.add_constant(X_probit_arr[:, keep_mask])
y_probit   = y_train.iloc[idx_sample].values
w_probit   = w_train.iloc[idx_sample].values
features_probit = [f for f, k in zip(cols_exo, keep_mask) if k]
print(f"  Muestra: {len(y_probit):,} obs | Features usadas en Probit: {X_probit.shape[1]-1}")

try:
    t0_probit = time.time()
    probit    = sm.Probit(y_probit, X_probit)
    res_probit = probit.fit(maxiter=150, disp=False)
    # McFadden R²
    mcfadden_probit = 1 - res_probit.llf / res_probit.llnull
    # Predicciones en test (mismas columnas filtradas)
    X_test_probit = sm.add_constant(X_test[cols_exo].fillna(0).values[:, keep_mask])
    y_pred_probit = res_probit.predict(X_test_probit)
    auc_probit = roc_auc_score(y_test, y_pred_probit)
    f1_probit  = f1_score(y_test, (y_pred_probit > 0.5).astype(int))
    print(f"  Probit ({elapsed(t0_probit)}) → AUC-ROC: {auc_probit:.4f}, "
          f"F1: {f1_probit:.4f}, McFadden R²: {mcfadden_probit:.4f}")
    probit_results = {
        "modelo":       "Probit baseline",
        "auc_roc":      round(auc_probit, 4),
        "f1":           round(f1_probit, 4),
        "mcfadden_r2":  round(mcfadden_probit, 4),
        "n_variables":  X_probit.shape[1] - 1
    }
    with open(MODELOS / "modelo_probit.pkl", "wb") as f:
        pickle.dump(res_probit, f)
except Exception as e:
    print(f"  ⚠️ Probit falló: {e}")
    probit_results = {"modelo": "Probit baseline", "auc_roc": None, "f1": None,
                      "mcfadden_r2": None, "n_variables": None}

# ══════════════════════════════════════════════════════════════════════════════
# 5.2 — LASSO Logístico con validación cruzada 5-fold (sklearn/saga)
# ══════════════════════════════════════════════════════════════════════════════
# Por qué sklearn en lugar de PyTorch:
#   LogisticRegressionCV con solver='saga' implementa LASSO (L1) de forma nativa,
#   con paralelización por n_jobs=-1 (todos los cores). Mucho más rápido y estable
#   que un loop manual en PyTorch para este tamaño de datos.
print("\n[5.2] Entrenando LASSO Logístico con CV 5-fold (sklearn saga)...")
print("  penalty='l1', cv=5, Cs=10, solver='saga', n_jobs=-1 (todos los cores)")
print("-" * 60)

import signal

X_np = X_train_s.fillna(0).values.astype(np.float32)
y_np = y_train.values
w_np = w_train.values.astype(np.float64)

# Normalizar pesos de encuesta a media=1 para preservar su estructura relativa
# sin crear magnitudes absolutas que desestabilicen SAGA.
# El balanceo de clases se delega a class_weight='balanced' dentro del modelo
# (sklearn lo implementa de forma numéricamente estable internamente).
w_survey = w_np / w_np.mean()

# 10 valores de C (suficiente para identificar la zona optima)
Cs = np.logspace(-3, 1, 10)

t0_lasso = time.time()
lasso_cv = LogisticRegressionCV(
    Cs=list(Cs),
    penalty="l1",
    solver="saga",
    cv=5,
    max_iter=2000,       # suficiente para convergencia con 439K obs y 54 features
    tol=1e-3,            # tolerancia estándar para SAGA con datasets grandes
    scoring="roc_auc",
    class_weight="balanced",  # sklearn balancea clases internamente (estable)
    n_jobs=-1,  # Usa todos los cores del servidor Lambda
    random_state=42,
    verbose=0,
)

# Bloquear SIGINT durante el entrenamiento: el entorno del servidor puede enviar
# SIGINT al proceso Python interrumpiendo el solver Cython (sag64). Lo bloqueamos
# temporalmente y lo restauramos al terminar.
_orig_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
print("  (SIGINT bloqueado durante entrenamiento para evitar interrupciones del servidor)")
try:
    lasso_cv.fit(X_np, y_np, sample_weight=w_survey)
finally:
    signal.signal(signal.SIGINT, _orig_sigint)  # restaurar siempre

best_C = lasso_cv.C_[0]
print(f"  ✅ CV completada en {elapsed(t0_lasso)}")
print(f"  C óptimo (1/λ): {best_C:.6f}  (λ = {1/best_C:.4f})")

# Medias AUC por C (para reporte)
mean_scores = lasso_cv.scores_[1].mean(axis=0)  # media sobre 5 folds
best_cv_auc = mean_scores.max()
print(f"  AUC-ROC CV promedio en C óptimo: {best_cv_auc:.4f}")

# ── Métricas en test ─────────────────────────────────────────────────────────
X_test_np    = X_test_s.fillna(0).values.astype(np.float32)
y_pred_prob  = lasso_cv.predict_proba(X_test_np)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype(int)
auc_lasso    = roc_auc_score(y_test, y_pred_prob)
f1_lasso     = f1_score(y_test, y_pred_class)

# McFadden R² para LASSO (comparación con modelo nulo)
p_bar    = y_train.mean()
# log-likelihood del modelo nulo (predicción = prevalencia)
ll_null  = len(y_test) * (
    p_bar * np.log(p_bar + 1e-15) + (1 - p_bar) * np.log(1 - p_bar + 1e-15)
)
# log-likelihood del modelo LASSO en test
ll_model = (
    y_test * np.log(np.clip(y_pred_prob, 1e-15, 1-1e-15)) +
    (1 - y_test) * np.log(np.clip(1 - y_pred_prob, 1e-15, 1-1e-15))
).sum()
mcfadden_lasso = 1 - ll_model / ll_null if ll_null != 0 else np.nan

print(f"\n  📊 Métricas LASSO en test:")
print(f"     AUC-ROC:     {auc_lasso:.4f}  (esperado: 0.74–0.79)")
print(f"     F1-score:    {f1_lasso:.4f}  (esperado: 0.70–0.75)")
print(f"     McFadden R²: {mcfadden_lasso:.4f}")
print(f"\n  {classification_report(y_test, y_pred_class, target_names=['Formal','Informal'])}")

# ── Variables seleccionadas (coef ≠ 0) ───────────────────────────────────────
coef = lasso_cv.coef_[0]
seleccionadas = pd.DataFrame({
    "feature":     feature_names,
    "coeficiente": coef,
}).query("coeficiente != 0").sort_values("coeficiente", key=abs, ascending=False)
print(f"  Variables seleccionadas (coef ≠ 0): {len(seleccionadas)} de {len(feature_names)}")
print(seleccionadas.head(20).to_string(index=False))
seleccionadas.to_csv(TABLAS / "tabla_coeficientes_lasso.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5.3 — Regularization Path plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5.3] Generando Regularization Path (15 valores de C)...")
Cs_path   = np.logspace(-4, 2, 15)   # 15 valores (reducido para mayor velocidad)
coefs_path = []
t0_path   = time.time()

# Mantener SIGINT bloqueado durante el path tambien
_orig_sigint2 = signal.signal(signal.SIGINT, signal.SIG_IGN)
try:
    for i, C in enumerate(Cs_path, 1):
        m = LogisticRegression(
            penalty="l1", solver="saga", C=C, max_iter=2000,
            tol=1e-3, class_weight="balanced", random_state=42, n_jobs=-1
        )
        m.fit(X_np, y_np, sample_weight=w_survey)
        coefs_path.append(m.coef_[0])
        pct = int(i / len(Cs_path) * 30)
        bar = "█" * pct + "░" * (30 - pct)
        print(f"  [{bar}] {i:2d}/15  C={C:.5f}  ({elapsed(t0_path)} transcurrido)",
              end="\r", flush=True)
finally:
    signal.signal(signal.SIGINT, _orig_sigint2)

print()
print(f"  ✅ Path calculado en {elapsed(t0_path)}")
coefs_path = np.array(coefs_path)

# Top 15 features por magnitud del coeficiente óptimo
top_idx = np.argsort(np.abs(coef))[-15:][::-1]

fig, ax = plt.subplots(figsize=(10, 6))
for i in top_idx:
    lbl = feature_names[i]
    lbl = lbl if len(lbl) < 30 else lbl[:27] + "..."
    ax.semilogx(Cs_path, coefs_path[:, i], linewidth=1.2, label=lbl)
ax.axvline(best_C, color="black", linewidth=1.5, linestyle="--",
           label=f"C óptimo = {best_C:.4f}")
ax.set_xlabel("C = 1/λ  (mayor C → menos regularización)")
ax.set_ylabel("Coeficiente")
ax.set_title("Regularization Path — LASSO Logístico\n(top 15 variables por magnitud del coeficiente)",
             fontsize=10)
ax.legend(fontsize=6, loc="upper left", ncol=2, frameon=False)
ax.axhline(0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.savefig(FIGS / "figura_lasso_regularization_path.pdf")
plt.savefig(FIGS / "figura_lasso_regularization_path.png", dpi=300)
plt.close()
print(f"  ✅ Regularization path guardado")

# ── Guardar modelo y resultados ───────────────────────────────────────────────
with open(MODELOS / "modelo_lasso.pkl", "wb") as f:
    pickle.dump(lasso_cv, f)

lasso_results = {
    "modelo":      "LASSO Logístico",
    "auc_roc":     round(auc_lasso, 4),
    "f1":          round(f1_lasso, 4),
    "mcfadden_r2": round(mcfadden_lasso, 4) if not np.isnan(mcfadden_lasso) else None,
    "n_variables": len(seleccionadas),
    "C_optimo":    round(float(best_C), 6),
}
with open(MODELOS / "resultados_lasso.pkl", "wb") as f:
    pickle.dump({"lasso": lasso_results, "probit": probit_results,
                 "y_pred_lasso": y_pred_prob}, f)

print(f"\n✅ PASO 5 COMPLETADO")
print(f"   AUC-ROC LASSO: {auc_lasso:.4f}")
print(f"   Variables seleccionadas: {len(seleccionadas)}")
print(f"   → modelos/modelo_lasso.pkl")
print(f"   → tablas/tabla_coeficientes_lasso.csv")
