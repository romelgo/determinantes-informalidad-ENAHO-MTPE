"""
Validación cruzada temporal (expanding window) con el conjunto de features
ENRIQUECIDO. Reconstruye la matriz X (igual que 04_features.py) conservando el
año, y reentrena XGBoost de forma incremental: train = años < t, test = año t.
Actualiza tablas/tabla_robustez_expanding_window.csv.
"""
import pandas as pd, numpy as np, pickle, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
import xgboost as xgb
warnings.filterwarnings("ignore")

from config import ROOT, RAW_ENAHO, INTERIM, ENAHO_FINAL, TAB
RAW  = RAW_ENAHO
FINAL= ENAHO_FINAL
PLAN = INTERIM / "planilla_limpia.csv"
CACHE= INTERIM / "enaho_enriquecido_cache.parquet"
YEARS= list(range(2015, 2026))

def ciiu_section(code):
    if pd.isna(code): return "NA"
    try: div=int(str(int(code)).zfill(4)[:2])
    except: return "NA"
    if 1<=div<=3: return "A_agro"
    elif 5<=div<=9: return "B_mineria"
    elif 10<=div<=33: return "C_manufactura"
    elif 35<=div<=39: return "DE_electr_agua"
    elif 41<=div<=43: return "F_construccion"
    elif 45<=div<=47: return "G_comercio"
    elif 49<=div<=53: return "H_transporte"
    elif 55<=div<=56: return "I_aloj_comida"
    elif 58<=div<=63: return "J_informacion"
    elif 64<=div<=66: return "K_financiero"
    elif div==68: return "L_inmobiliario"
    elif 69<=div<=82: return "MN_profes_admin"
    elif div==84: return "O_adm_publica"
    elif div==85: return "P_ensenanza"
    elif 86<=div<=88: return "Q_salud"
    elif 90<=div<=96: return "RS_arte_otros"
    elif 97<=div<=98: return "T_hogares"
    else: return "U_otros"

def extraer(year):
    f=RAW/f"Enaho01a-{year}-500.csv"
    first=open(f,encoding="latin1",errors="replace").readline()
    sep=";" if first.count(";")>first.count(",") else ","
    df=pd.read_csv(f,encoding="latin1",sep=sep,low_memory=False,on_bad_lines="skip")
    df.columns=[c.upper().strip() for c in df.columns]
    df["P500I"]=df["P500I"].astype(str).str.strip().str.zfill(2); df=df[df["P500I"]!="00"]
    for c in ["P204","P205","P206","OCU500","P208A","P207"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    res=((df.P204==1)&(df.P205==2))|((df.P204==2)&(df.P206==1))
    df=df[res&(df.OCU500<3)&(df.P208A.between(14,98))].copy()
    df["SEXO"]=df["P207"]; df["EDAD"]=df["P208A"]
    for c in ["CONGLOME","VIVIENDA","HOGAR"]: df[c]=df[c].astype(str).str.strip()
    df["catocup"]=pd.to_numeric(df["P507"],errors="coerce")
    df["tam_emp"]=pd.to_numeric(df["P512A"],errors="coerce")
    df["horas"]=pd.to_numeric(df["P513T"],errors="coerce")
    ing=pd.to_numeric(df.get("D529T",np.nan),errors="coerce"); df["log_ing"]=np.log1p(ing.clip(lower=0))
    df["sector"]=pd.to_numeric(df["P506R4"],errors="coerce").map(ciiu_section)  # P506R4 = rama CIIU (P505R4 es ocupación)
    key=["CONGLOME","VIVIENDA","HOGAR","SEXO","EDAD"]
    out=df[key+["catocup","tam_emp","horas","log_ing","sector"]].drop_duplicates(key,keep=False)
    out["AÑO"]=year; return out

if CACHE.exists():
    df=pd.read_parquet(CACHE)
    print(f"Cache cargada: {df.shape}")
else:
    print("Extrayendo crudos (11 años)...")
    ricos=pd.concat([extraer(y) for y in YEARS],ignore_index=True)
    fin=pd.read_csv(FINAL,low_memory=False)
    fin.columns=[c.strip().upper().replace("﻿","") for c in fin.columns]
    fin.rename(columns={"DEP":"DEPARTAMENTO"},inplace=True)
    fin["Y"]=pd.to_numeric(fin["TEI"],errors="coerce"); fin=fin[fin["Y"].isin([0,1])].copy()
    for c in ["CONGLOME","VIVIENDA","HOGAR"]: fin[c]=fin[c].astype(str).str.strip()
    fin["SEXO"]=pd.to_numeric(fin["SEXO"],errors="coerce"); fin["EDAD"]=pd.to_numeric(fin["EDAD"],errors="coerce")
    df=fin.merge(ricos,on=["AÑO","CONGLOME","VIVIENDA","HOGAR","SEXO","EDAD"],how="inner")
    plan=pd.read_csv(PLAN); plan.columns=[c.strip().upper().replace("﻿","") for c in plan.columns]
    pv=[v for v in ["PCT_MUJER_FORMAL","PCT_JOVEN_FORMAL","PCT_INDEFINIDO_FORMAL","PCT_MYPE_FORMAL","PCT_NO_CALIFICADO_FORMAL"] if v in plan.columns]
    df=df.merge(plan[["AÑO"]+pv].rename(columns={v:v.lower() for v in pv}),on="AÑO",how="left")
    df.to_parquet(CACHE); print(f"Cache guardada: {df.shape}")

# Construir X (mismo encoding que 04)
df["grupo_edad"]=np.where(df.EDAD.between(14,29),1,np.where(df.EDAD.between(30,59),2,3))
df["sexo"]=df["SEXO"].map({1:0,2:1}); df["area"]=pd.to_numeric(df["AREA"],errors="coerce").map({1:0,2:1})
df["edad"]=df["EDAD"].astype(float); df["region"]=pd.to_numeric(df["REGION"],errors="coerce")
df["departamento"]=pd.to_numeric(df["DEPARTAMENTO"],errors="coerce"); df["nivel_edu"]=pd.to_numeric(df["NIVEL_EDU"],errors="coerce")
df["anio"]=df["AÑO"].astype(int); df["anio_covid"]=df["anio"].isin([2020,2021]).astype(int)
df["mype"]=(df["tam_emp"]<=1).astype(float); df["horas"]=df["horas"].fillna(df["horas"].median()); df["log_ing"]=df["log_ing"].fillna(0.0)
plan_low=[c for c in ["pct_mujer_formal","pct_joven_formal","pct_indefinido_formal","pct_mype_formal","pct_no_calificado_formal"] if c in df.columns]
num=["sexo","area","edad","anio_covid","horas","log_ing","mype"]+plan_low
cat=["grupo_edad","region","departamento","nivel_edu","catocup","tam_emp","sector"]
base=df[num+cat+["Y","anio"]].dropna(subset=["Y"]).reset_index(drop=True)
Xenc=pd.get_dummies(base,columns=cat,drop_first=True,dtype=float)
anio=base["anio"].values; y=base["Y"].astype(int).values
feat=[c for c in Xenc.columns if c not in ["Y","anio"]]
X=Xenc[feat].fillna(0).values
print(f"X enriquecido: {X.shape}")

rows=[]
for yp in range(2019,2026):
    tr=anio<yp; te=anio==yp
    if te.sum()==0 or tr.sum()==0: continue
    n_pos=int(y[tr].sum()); n_neg=int((1-y[tr]).sum())
    m=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.85,
                        colsample_bytree=0.85,scale_pos_weight=n_neg/max(n_pos,1),
                        eval_metric="auc",n_jobs=-1,verbosity=0)
    m.fit(X[tr],y[tr])
    auc=roc_auc_score(y[te],m.predict_proba(X[te])[:,1])
    rows.append({"año_pred":yp,"auc_roc":round(auc,4),"n_train":int(tr.sum()),"n_test":int(te.sum())})
    print(f"  {yp}: AUC={auc:.4f} (train={tr.sum():,}, test={te.sum():,})")

out=pd.DataFrame(rows)
out.to_csv(TAB/"tabla_robustez_expanding_window.csv",index=False)
print("\n✅ tabla_robustez_expanding_window.csv actualizada")
print(out.to_string(index=False))
