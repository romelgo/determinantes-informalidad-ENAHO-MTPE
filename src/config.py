"""
config.py — Rutas centralizadas del proyecto.

Todos los scripts del pipeline importan las rutas desde aquí:

    from config import RAW_ENAHO, RAW_PLANILLA, INTERIM, PROCESSED, MODELS, FIG, TAB

ROOT se deduce de la ubicación de este archivo (src/config.py → raíz del repo),
así el pipeline funciona sin importar el directorio de trabajo ni el usuario.
"""

from pathlib import Path

# Raíz del repositorio = carpeta padre de src/
ROOT = Path(__file__).resolve().parent.parent

# ── Datos ─────────────────────────────────────────────────────────────────────
DATA         = ROOT / "data"
RAW          = DATA / "raw"
RAW_ENAHO    = RAW / "enaho01a-500"     # 11 CSV crudos módulo 500 (2015-2025)
RAW_PLANILLA = RAW / "planilla"         # 6 CSV Planilla Electrónica SUNAT/MTPE
INTERIM      = DATA / "interim"         # CSV intermedios (final ENAHO, planilla_limpia, integrado)
PROCESSED    = DATA / "processed"       # .pkl listos para modelar (X/y/w, scaler)

# Atajos a archivos intermedios usados por varios scripts
ENAHO_FINAL  = INTERIM / "enaho_empleo_2015_2025_final.csv"

# ── Salidas ───────────────────────────────────────────────────────────────────
MODELS  = ROOT / "models"               # modelos entrenados (.pkl)
OUTPUTS = ROOT / "outputs"
FIG     = OUTPUTS / "figures"           # figuras (PNG/PDF)
TAB     = OUTPUTS / "tables"            # tablas (CSV/TEX)

# Crear carpetas de salida si no existen (idempotente)
for _d in (INTERIM, PROCESSED, MODELS, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)
