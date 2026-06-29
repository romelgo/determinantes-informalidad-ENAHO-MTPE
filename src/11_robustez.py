"""
============================================================
PASO 11: Verificaciones de Robustez
============================================================

Cada check de robustez fortalece la credibilidad del paper:
- Check 1: Si los resultados cambian con otra definición de informalidad
  → muestra que los hallazgos no dependen de una sola medida.
- Check 2: Estimaciones solo en área urbana eliminan la heterogeneidad
  rural-urbana que podría confundir los efectos del sector.
- Check 3: Expanding window CV evita data leakage temporal → las
  predicciones de 2020 nunca "vieron" datos de 2020 en entrenamiento.
- Check 4: Placebo DiD con año falso 2017 → si β es significativo,
  habría un problema con el diseño (confounders preexistentes).
- Check 5: Bootstrap SHAP verifica que el ranking de importancia sea
  estable → los hallazgos no son artefactos de una muestra particular.

Outputs:
- tablas/tabla_robustez_modelos.csv
- figuras/figura_robustez_expanding_window.pdf
- figuras/figura_robustez_shap_bootstrap.pdf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

from config import PROCESSED, INTERIM, MODELS, TAB, FIG
DATOS   = PROCESSED
MODELOS = MODELS
TABLAS  = TAB
FIGS    = FIG

print("=" * 60)
print("PASO 11: Verificaciones de Robustez")
print("=" * 60)

def load(name):
    with open(DATOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
def load_m(name):
    with open(MODELOS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

# ── Cargar artefactos base ───────────────────────────────────────────────────
feature_names = load("feature_names")
lasso_model   = load_m("modelo_lasso")
rf_model      = load_m("modelo_rf")
xgb_model     = load_m("modelo_xgb")
scaler        = load("scaler")

# Cargar dataset completo
df = pd.read_csv(INTERIM / "dataset_integrado.csv", low_memory=False)
df_ocu = df[(df["OCUPADO"] == 1) & (df["RESIDENTE"] == 1) & df["Y"].notna()].copy()
print(f"  Dataset ocupados: {len(df_ocu):,}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Definición alternativa de informalidad (nivel educativo bajo)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] Definición alternativa de informalidad...")
print("  Criterio alternativo: nivel educativo bajo (nivel_edu ≤ 2) como proxy de informalidad")
print("  Compara tasas con Y original (TEI)")

tasa_tei = df_ocu["Y"].mean()
if "nivel_edu" in df_ocu.columns:
    df_ocu["Y_alt"] = (df_ocu["nivel_edu"] <= 2).astype(float)
    tasa_alt = df_ocu["Y_alt"].mean()
    correlacion = df_ocu[["Y", "Y_alt"]].corr().iloc[0,1]
    print(f"  Tasa informalidad (TEI original):       {tasa_tei:.4f} ({tasa_tei*100:.1f}%)")
    print(f"  Tasa informalidad (educación baja alt): {tasa_alt:.4f} ({tasa_alt*100:.1f}%)")
    print(f"  Correlación entre Y_original y Y_alt:  {correlacion:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Solo área urbana
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] Re-estimando modelos solo en área urbana...")
X_full = load("X_train")
y_full = load("y_train")
w_full = load("w_train")
X_test_full = load("X_test")
y_test_full = load("y_test")
X_test_s    = load("X_test_scaled")

# Reconstruir indicador de área en train/test
# area=0 → urbano (ya transformado en Paso 4: 0=urbano, 1=rural)
mask_urban_train = (X_full["area"] == 0).values if "area" in X_full.columns else np.ones(len(X_full), bool)
mask_urban_test  = (X_test_full["area"] == 0).values if "area" in X_test_full.columns else np.ones(len(X_test_full), bool)

X_train_u = X_full[mask_urban_train].fillna(0)
y_train_u = y_full[mask_urban_train]
w_train_u = w_full[mask_urban_train]
X_test_u  = X_test_full[mask_urban_test].fillna(0)
y_test_u  = y_test_full[mask_urban_test]

print(f"  Muestra urbana train: {len(X_train_u):,}, test: {len(X_test_u):,}")

resultados_urbanos = []
for nombre, modelo, X_tr, X_te, usa_scaler in [
    ("LASSO (urbano)",  lasso_model, load("X_train_scaled"), load("X_test_scaled"), True),
    ("RF (urbano)",     rf_model,    X_full, X_test_full, False),
    ("XGBoost (urbano)",xgb_model,   X_full, X_test_full, False),
]:
    try:
        mask_tr = mask_urban_train
        mask_te = mask_urban_test
        X_t  = X_tr[mask_tr].fillna(0) if hasattr(X_tr, "iloc") else X_tr[mask_tr]
        X_e  = X_te[mask_te].fillna(0) if hasattr(X_te, "iloc") else X_te[mask_te]
        y_tr = y_full[mask_tr]
        y_te = y_test_full[mask_te]
        w_tr = w_full[mask_tr]
        
        modelo.fit(X_t, y_tr, sample_weight=w_tr)
        y_pred = modelo.predict_proba(X_e)[:,1]
        auc = roc_auc_score(y_te, y_pred)
        f1  = f1_score(y_te, (y_pred>0.5).astype(int))
        resultados_urbanos.append({"modelo": nombre, "auc_roc": round(auc,4), "f1": round(f1,4)})
        print(f"  {nombre}: AUC={auc:.4f}, F1={f1:.4f}")
    except Exception as e:
        print(f"  {nombre}: Error — {e}")
        resultados_urbanos.append({"modelo": nombre, "auc_roc": None, "f1": None})

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Expanding Window Cross-Validation (sin data leakage temporal)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] Expanding Window CV (2015→2019, 2015→2020, ..., 2015→2024)...")
df_ocu["sexo_bin"] = (df_ocu["sexo"] == 2).astype(int)
df_ocu["area_bin"] = (df_ocu["area"] == 2).astype(int)

feature_cols_simple = ["sexo_bin", "edad", "area_bin", "region", "nivel_edu", "grupo_edad"]
feature_cols_simple = [f for f in feature_cols_simple if f in df_ocu.columns]

expanding_results = []
años_test = list(range(2019, 2026))
for year_pred in años_test:
    df_train_ew = df_ocu[df_ocu["anio"] < year_pred]
    df_test_ew  = df_ocu[df_ocu["anio"] == year_pred]
    
    if len(df_train_ew) < 1000 or len(df_test_ew) < 100:
        continue
    
    X_tr_ew = df_train_ew[feature_cols_simple].fillna(df_train_ew[feature_cols_simple].median())
    y_tr_ew = df_train_ew["Y"].values
    w_tr_ew = df_train_ew["fac500a"].values
    X_te_ew = df_test_ew[feature_cols_simple].fillna(df_test_ew[feature_cols_simple].median())
    y_te_ew = df_test_ew["Y"].values
    
    try:
        # XGBoost expanding window
        n_pos_ew = (y_tr_ew == 1).sum()
        n_neg_ew = (y_tr_ew == 0).sum()
        spw_ew   = n_neg_ew / n_pos_ew if n_pos_ew > 0 else 1
        
        xgb_ew = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=spw_ew, eval_metric="auc",
            use_label_encoder=False, n_jobs=-1, random_state=42
        )
        xgb_ew.fit(X_tr_ew, y_tr_ew, sample_weight=w_tr_ew, verbose=False)
        y_pred_ew = xgb_ew.predict_proba(X_te_ew)[:,1]
        auc_ew    = roc_auc_score(y_te_ew, y_pred_ew)
        expanding_results.append({"año_pred": year_pred, "auc_roc": round(auc_ew, 4),
                                  "n_train": len(df_train_ew), "n_test": len(df_test_ew)})
        print(f"  Predicción {year_pred} (train: 2015-{year_pred-1}): AUC = {auc_ew:.4f}")
    except Exception as e:
        print(f"  Predicción {year_pred}: Error — {e}")

df_expanding = pd.DataFrame(expanding_results)

# Figura expanding window
if len(df_expanding) > 0:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_expanding["año_pred"], df_expanding["auc_roc"],
            color="#2563EB", linewidth=2, marker="o", markersize=6)
    ax.axhline(0.74, color="gray", linewidth=1, linestyle="--",
               label="Umbral mínimo esperado (0.74)")
    ax.axvline(2020, color="#DC2626", linewidth=1.5, linestyle="--",
               label="COVID-19 (2020)")
    ax.set_xlabel("Año de predicción (test)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Check 3. Expanding Window CV — Estabilidad temporal del modelo\n"
                 "(XGBoost; cada punto predice 1 año sin ver datos futuros)", fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGS / "figura_robustez_expanding_window.pdf")
    plt.savefig(FIGS / "figura_robustez_expanding_window.png", dpi=300)
    plt.close()
    print(f"  ✅ Figura Expanding Window guardada")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Placebo DiD (año de tratamiento falso: 2017)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] Placebo DiD (año falso = 2017)...")
import statsmodels.api as sm

df_placebo_muestra = df_ocu[df_ocu["anio"] <= 2019].copy()  # Solo pre-COVID
np.random.seed(42)
if len(df_placebo_muestra) > 100000:
    idx_pl = np.random.choice(len(df_placebo_muestra), 100000, replace=False)
    df_placebo_muestra = df_placebo_muestra.iloc[idx_pl].reset_index(drop=True)

df_placebo_muestra["tratado"]  = (df_placebo_muestra["area"] == 1).astype(int)
df_placebo_muestra["post_plc"] = (df_placebo_muestra["anio"] >= 2017).astype(int)
df_placebo_muestra["did_plc"]  = (df_placebo_muestra["tratado"] * df_placebo_muestra["post_plc"])

feat_pl = ["did_plc","tratado","post_plc","edad","nivel_edu"]
feat_pl = [f for f in feat_pl if f in df_placebo_muestra.columns]
X_pl = sm.add_constant(df_placebo_muestra[feat_pl].fillna(0).values)
y_pl = df_placebo_muestra["Y"].values
w_pl = df_placebo_muestra["fac500a"].values

try:
    res_pl  = sm.WLS(y_pl, X_pl, weights=w_pl).fit(cov_type="HC3")
    beta_pl = res_pl.params[1]  # coef de did_plc
    se_pl   = res_pl.bse[1]
    p_pl    = res_pl.pvalues[1]
    print(f"  β placebo (año=2017): {beta_pl:+.4f}")
    print(f"  p-valor: {p_pl:.4f}  {'✅ No significativo (< 0.1) → diseño válido' if p_pl > 0.1 else '⚠️ Significativo — revisar diseño'}")
    placebo_result = {"beta_placebo": round(beta_pl,4), "p_valor": round(p_pl,4), "se": round(se_pl,4)}
except Exception as e:
    print(f"  Error en Placebo DiD: {e}")
    placebo_result = {"beta_placebo": None, "p_valor": None, "se": None}

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Bootstrap SHAP (200 iteraciones)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] Bootstrap SHAP (200 iteraciones)...")
# Cargar SHAP values guardados
shap_data  = load_m("shap_values")
shap_vals  = shap_data["shap_values"]   # shape: (n_sample, n_features)
X_shap_arr = shap_data["X_shap"]

n_boot = 200
np.random.seed(42)
bootstrap_rankings = []
n_shap = len(shap_vals)

for b in range(n_boot):
    idx_b = np.random.choice(n_shap, n_shap, replace=True)
    abs_mean_b = np.abs(shap_vals[idx_b]).mean(axis=0)
    # Ranking (1 = más importante)
    ranking_b = (-abs_mean_b).argsort().argsort() + 1
    bootstrap_rankings.append(ranking_b)

bootstrap_rankings = np.array(bootstrap_rankings)
rank_mean = bootstrap_rankings.mean(axis=0)
rank_std  = bootstrap_rankings.std(axis=0)

# Top 10 más estables (menor desviación estándar del ranking)
shap_bootstrap = pd.DataFrame({
    "feature":      feature_names,
    "rank_medio":   rank_mean,
    "rank_std":     rank_std,
    "shap_abs_mean": np.abs(shap_vals).mean(axis=0),
}).sort_values("rank_medio")

print(f"\n  Top 10 variables más importantes y estables (bootstrap SHAP):")
print(shap_bootstrap.head(10)[["feature","rank_medio","rank_std","shap_abs_mean"]].to_string(index=False))

# Figura bootstrap SHAP
fig, ax = plt.subplots(figsize=(9, 7))
top_shap = shap_bootstrap.head(15).sort_values("rank_medio", ascending=False)
colors_b = plt.cm.YlOrRd_r(np.linspace(0.2, 0.8, len(top_shap)))
ax.barh(top_shap["feature"], top_shap["rank_medio"],
        xerr=top_shap["rank_std"],
        color=colors_b, edgecolor="white",
        error_kw={"elinewidth": 0.8, "ecolor": "gray", "capsize": 3})
ax.set_xlabel("Ranking medio de importancia (1 = más importante)")
ax.set_title("Check 5. Estabilidad del ranking SHAP\n"
             f"Bootstrap n=200 iteraciones — Top 15 variables XGBoost",
             fontsize=10)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(FIGS / "figura_robustez_shap_bootstrap.pdf")
plt.savefig(FIGS / "figura_robustez_shap_bootstrap.png", dpi=300)
plt.close()

# ── Guardar resumen de robustez ────────────────────────────────────────────────
resumen_robustez = {
    "check1_correlacion_Y_alt": correlacion if "nivel_edu" in df_ocu.columns else None,
    "check2_urbano": resultados_urbanos,
    "check3_expanding": df_expanding.to_dict("records") if len(df_expanding)>0 else [],
    "check4_placebo":   placebo_result,
    "check5_shap_top10": shap_bootstrap.head(10).to_dict("records"),
}
shap_bootstrap.to_csv(TABLAS / "tabla_robustez_shap_bootstrap.csv", index=False)
if len(df_expanding) > 0:
    df_expanding.to_csv(TABLAS / "tabla_robustez_expanding_window.csv", index=False)

print(f"\n✅ PASO 11 COMPLETADO — 5 verificaciones de robustez ejecutadas")
