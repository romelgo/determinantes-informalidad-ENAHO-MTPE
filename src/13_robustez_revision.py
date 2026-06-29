#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_robustez_revision.py — Verificaciones de robustez solicitadas por los revisores.

(A) Sensibilidad de la definición de la variable objetivo: compara el indicador
    OCUPINF (INEI, disponible 2015-2023) contra el proxy compuesto TEI
    (P511A/P510A1/P510B, usado para 2024-2025) en los años donde AMBOS pueden
    construirse, cuantificando acuerdo, kappa de Cohen y diferencia en la tasa de
    informalidad estimada.

(B) Robustez del DiD COVID-19: re-estima el efecto usando como grupo tratado los
    SECTORES CIIU de alto contacto (en vez del proxy 'área urbana'), para validar
    que el efecto no es artefacto del proxy.

Salidas: tablas/tabla_robustez_definicion.csv, tablas/tabla_robustez_did_ciiu.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
RAW   = ROOT / "data" / "raw" / "enaho01a-500"
FINAL = ROOT / "data" / "interim" / "enaho_empleo_2015_2025_final.csv"
TAB   = ROOT / "outputs" / "tables"
YEARS = list(range(2015, 2026))


def read_raw(year, usecols):
    f = RAW / f"Enaho01a-{year}-500.csv"
    first = open(f, encoding="latin1", errors="replace").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(f, encoding="latin1", sep=sep, low_memory=False,
                     on_bad_lines="skip")
    df.columns = [c.upper().strip() for c in df.columns]
    keep = [c for c in usecols if c in df.columns]
    return df[keep].copy()


def pop_filter(df):
    """Filtros de población idénticos al pipeline (PEA ocupada, residente, edad)."""
    df["P500I"] = df["P500I"].astype(str).str.strip().str.zfill(2)
    df = df[df["P500I"] != "00"]
    for c in ["P204", "P205", "P206", "OCU500", "P208A"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    res = ((df.P204 == 1) & (df.P205 == 2)) | ((df.P204 == 2) & (df.P206 == 1))
    return df[res & (df.OCU500 < 3) & (df.P208A.between(14, 98))].copy()


def tei_proxy(df):
    """Proxy compuesto: informal si NO contrato (P511A=7) o NO RUC (P510A1=3)
    o NO libros (P510B=2). Replica analisis_enaho/fix_notebooks.py."""
    a1 = df.get("P510A1", pd.Series("", index=df.index)).astype(str).str.strip()
    b  = df.get("P510B",  pd.Series("", index=df.index)).astype(str).str.strip()
    c  = df.get("P511A",  pd.Series("", index=df.index)).astype(str).str.strip()
    a1_inf, a1_ok = (a1 == "3"), a1.isin(["1", "2", "3"])
    b_inf,  b_ok  = (b == "2"),  b.isin(["1", "2"])
    c_inf,  c_ok  = (c == "7"),  c.isin(["1", "2", "3", "4", "5", "6", "7"])
    any_ok  = a1_ok | b_ok | c_ok
    any_inf = a1_inf | b_inf | c_inf
    return np.where(any_ok, any_inf.astype(float), np.nan)


# ════════════════════════════════════════════════════════════════════════════
# (A) Sensibilidad de la definición de informalidad
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("(A) SENSIBILIDAD: OCUPINF (INEI) vs proxy TEI compuesto")
print("=" * 70)

cols_def = ["P500I", "P204", "P205", "P206", "OCU500", "P208A",
            "OCUPINF", "P510A1", "P510B", "P511A", "FAC500A"]
rows = []
conf_rows = []                       # matriz de concordancia para años representativos
AÑOS_CONF = [2019, 2021]             # pre-COVID y COVID
for y in range(2015, 2024):          # años con OCUPINF disponible
    df = pop_filter(read_raw(y, cols_def))
    df["ocupinf_inf"] = np.where(
        pd.to_numeric(df["OCUPINF"], errors="coerce").notna(),
        (pd.to_numeric(df["OCUPINF"], errors="coerce") == 1).astype(float), np.nan)
    df["tei_inf"] = tei_proxy(df)
    w = pd.to_numeric(df["FAC500A"], errors="coerce")
    both = df[df["ocupinf_inf"].notna() & df["tei_inf"].notna()].copy()
    wb = pd.to_numeric(both["FAC500A"], errors="coerce").values
    o, t = both["ocupinf_inf"].values, both["tei_inf"].values
    rate_o = np.average(o, weights=wb) * 100
    rate_t = np.average(t, weights=wb) * 100
    agree  = np.average((o == t).astype(float), weights=wb) * 100
    # kappa de Cohen ponderado
    po = agree / 100
    pe = (np.average(o, weights=wb) * np.average(t, weights=wb) +
          (1 - np.average(o, weights=wb)) * (1 - np.average(t, weights=wb)))
    kappa = (po - pe) / (1 - pe)
    rows.append({"anio": y, "N": len(both),
                 "tasa_OCUPINF": round(rate_o, 1),
                 "tasa_TEIproxy": round(rate_t, 1),
                 "dif_pp": round(rate_t - rate_o, 2),
                 "acuerdo_pct": round(agree, 1),
                 "kappa": round(kappa, 3)})
    print(f"  {y}: N={len(both):>6,}  OCUPINF={rate_o:5.1f}%  "
          f"TEI={rate_t:5.1f}%  dif={rate_t-rate_o:+5.2f}pp  "
          f"acuerdo={agree:5.1f}%  kappa={kappa:.3f}")

    # ── Matriz de concordancia OCUPINF (verdad) vs TEI (proxy) ────────────────
    if y in AÑOS_CONF:
        tp = np.average(((o == 1) & (t == 1)).astype(float), weights=wb)
        tn = np.average(((o == 0) & (t == 0)).astype(float), weights=wb)
        fp = np.average(((o == 0) & (t == 1)).astype(float), weights=wb)
        fn = np.average(((o == 1) & (t == 0)).astype(float), weights=wb)
        sens = tp / (tp + fn)                      # sensibilidad (recall informales)
        spec = tn / (tn + fp)                      # especificidad
        vpp  = tp / (tp + fp)                      # valor predictivo positivo
        vpn  = tn / (tn + fn)                      # valor predictivo negativo
        conf_rows.append({"anio": y, "VP": round(tp, 4), "VN": round(tn, 4),
                          "FP": round(fp, 4), "FN": round(fn, 4),
                          "sensibilidad": round(sens, 3), "especificidad": round(spec, 3),
                          "VPP": round(vpp, 3), "VPN": round(vpn, 3)})
        print(f"      [matriz {y}] Sens={sens:.3f}  Espec={spec:.3f}  "
              f"VPP={vpp:.3f}  VPN={vpn:.3f}")

tab_def = pd.DataFrame(rows)
tot = {"anio": "Global (2015-2023)", "N": tab_def["N"].sum(),
       "tasa_OCUPINF": round(tab_def["tasa_OCUPINF"].mean(), 1),
       "tasa_TEIproxy": round(tab_def["tasa_TEIproxy"].mean(), 1),
       "dif_pp": round(tab_def["dif_pp"].mean(), 2),
       "acuerdo_pct": round(tab_def["acuerdo_pct"].mean(), 1),
       "kappa": round(tab_def["kappa"].mean(), 3)}
tab_def = pd.concat([tab_def, pd.DataFrame([tot])], ignore_index=True)
tab_def.to_csv(TAB / "tabla_robustez_definicion.csv", index=False)
print(f"\n  -> {TAB/'tabla_robustez_definicion.csv'}")

# Matriz de concordancia OCUPINF–TEI (años representativos)
pd.DataFrame(conf_rows).to_csv(TAB / "tabla_concordancia_ocupinf_tei.csv", index=False)
print(f"  -> {TAB/'tabla_concordancia_ocupinf_tei.csv'}")


# ════════════════════════════════════════════════════════════════════════════
# (B) DiD con sectores CIIU de alto contacto
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("(B) ROBUSTEZ DiD: tratado = sectores CIIU de alto contacto")
print("=" * 70)

def ciiu_section(code):
    if pd.isna(code):
        return "NA"
    try:
        div = int(str(int(code)).zfill(4)[:2])
    except (ValueError, TypeError):
        return "NA"
    if   1 <= div <= 3:   return "A_agro"
    elif 5 <= div <= 9:   return "B_mineria"
    elif 10 <= div <= 33: return "C_manufactura"
    elif 35 <= div <= 39: return "DE_electr_agua"
    elif 41 <= div <= 43: return "F_construccion"
    elif 45 <= div <= 47: return "G_comercio"
    elif 49 <= div <= 53: return "H_transporte"
    elif 55 <= div <= 56: return "I_aloj_comida"
    elif 58 <= div <= 63: return "J_informacion"
    elif 64 <= div <= 66: return "K_financiero"
    elif div == 68:       return "L_inmobiliario"
    elif 69 <= div <= 82: return "MN_profes_admin"
    elif div == 84:       return "O_adm_publica"
    elif div == 85:       return "P_ensenanza"
    elif 86 <= div <= 88: return "Q_salud"
    elif 90 <= div <= 96: return "RS_arte_otros"
    elif 97 <= div <= 98: return "T_hogares"
    else:                 return "U_otros"

# Servicios presenciales de alto contacto (mayor exposición a restricciones COVID)
HIGH_CONTACT = {"G_comercio", "H_transporte", "I_aloj_comida",
                "RS_arte_otros", "T_hogares"}

# IMPORTANTE: la rama de actividad económica (CIIU rev.4) es P506R4
# ("¿a qué se dedica el negocio/empresa?", revisión 4). NO confundir con P505R4,
# que es la OCUPACIÓN principal (CNO-2015). Aquí necesitamos el SECTOR → P506R4.
# sector a nivel trabajador desde los crudos (clave de merge igual que 04)
cols_sec = ["P500I", "P204", "P205", "P206", "OCU500", "P208A", "P207",
            "CONGLOME", "VIVIENDA", "HOGAR", "P506R4"]
sec_list = []
for y in YEARS:
    df = pop_filter(read_raw(y, cols_sec))
    df["SEXO"] = pd.to_numeric(df["P207"], errors="coerce")
    df["EDAD"] = pd.to_numeric(df["P208A"], errors="coerce")
    for c in ["CONGLOME", "VIVIENDA", "HOGAR"]:
        df[c] = df[c].astype(str).str.strip()
    df["sector"] = pd.to_numeric(df["P506R4"], errors="coerce").map(ciiu_section)
    key = ["CONGLOME", "VIVIENDA", "HOGAR", "SEXO", "EDAD"]
    out = df[key + ["sector"]].drop_duplicates(key, keep=False)
    out["AÑO"] = y
    sec_list.append(out)
    print(f"   sector {y}: {len(out):,} claves únicas")
sec = pd.concat(sec_list, ignore_index=True)

fin = pd.read_csv(FINAL, low_memory=False)
fin.columns = [c.strip().upper().replace("﻿", "") for c in fin.columns]
fin["Y"] = pd.to_numeric(fin["TEI"], errors="coerce")
fin = fin[fin["Y"].isin([0, 1])].copy()
for c in ["CONGLOME", "VIVIENDA", "HOGAR"]:
    fin[c] = fin[c].astype(str).str.strip()
fin["SEXO"] = pd.to_numeric(fin["SEXO"], errors="coerce")
fin["EDAD"] = pd.to_numeric(fin["EDAD"], errors="coerce")
fin["AÑO"] = pd.to_numeric(fin["AÑO"], errors="coerce")

df = fin.merge(sec, on=["AÑO", "CONGLOME", "VIVIENDA", "HOGAR", "SEXO", "EDAD"],
               how="inner")
df = df[df["sector"].notna() & (df["sector"] != "NA")].copy()
df["tratado"] = df["sector"].isin(HIGH_CONTACT).astype(int)
df["post"]    = (df["AÑO"] >= 2020).astype(int)
df["did"]     = df["tratado"] * df["post"]
df["edad"]    = pd.to_numeric(df["EDAD"], errors="coerce")
df["sexo_m"]  = (pd.to_numeric(df["SEXO"], errors="coerce") == 2).astype(float)
df["nivel_edu"] = pd.to_numeric(df["NIVEL_EDU"], errors="coerce")
df["gedad"]   = pd.to_numeric(df["GEDAD"], errors="coerce")
df["w"]       = pd.to_numeric(df["FAC500A"], errors="coerce")
df["dep"]     = pd.to_numeric(df["DEP"], errors="coerce")

controls = ["edad", "sexo_m", "nivel_edu", "gedad"]
mdl = df[["Y", "tratado", "post", "did", "w", "dep"] + controls].dropna()
X = sm.add_constant(mdl[["tratado", "post", "did"] + controls].astype(float))
res = sm.WLS(mdl["Y"], X, weights=mdl["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": mdl["dep"]})

print(f"\n  N={len(mdl):,}  tratados(alto contacto)="
      f"{mdl['tratado'].mean()*100:.1f}%")
beta = res.params["did"]
print(f"  beta DiD (Post x AltoContacto) = {beta:+.4f}  "
      f"SE={res.bse['did']:.4f}  t={res.tvalues['did']:.2f}  "
      f"p={res.pvalues['did']:.4g}")

out = pd.DataFrame({
    "variable": ["Constante", "Tratado (alto contacto)", "Post (>=2020)",
                 "DiD (Post x Tratado)", "Edad", "Sexo (mujer=1)",
                 "Nivel educativo", "Grupo de edad"],
    "coef": [res.params["const"], res.params["tratado"], res.params["post"],
             res.params["did"], res.params["edad"], res.params["sexo_m"],
             res.params["nivel_edu"], res.params["gedad"]],
    "se":   [res.bse["const"], res.bse["tratado"], res.bse["post"],
             res.bse["did"], res.bse["edad"], res.bse["sexo_m"],
             res.bse["nivel_edu"], res.bse["gedad"]],
    "t":    [res.tvalues["const"], res.tvalues["tratado"], res.tvalues["post"],
             res.tvalues["did"], res.tvalues["edad"], res.tvalues["sexo_m"],
             res.tvalues["nivel_edu"], res.tvalues["gedad"]],
    "p":    [res.pvalues["const"], res.pvalues["tratado"], res.pvalues["post"],
             res.pvalues["did"], res.pvalues["edad"], res.pvalues["sexo_m"],
             res.pvalues["nivel_edu"], res.pvalues["gedad"]],
})
out = out.round(4)
out.to_csv(TAB / "tabla_robustez_did_ciiu.csv", index=False)
print(f"\n  -> {TAB/'tabla_robustez_did_ciiu.csv'}")
print("\nLISTO.")
