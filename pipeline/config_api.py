import requests
from pathlib import Path

# ===========================
# Configuración API RESFULL
# ===========================

API_ROOT = "http://127.0.0.1:8000"
API_PREFIX = "/api"

# Endpoints base (ajusta nombres según tu API real)
BASE_INGESTAS      = f"{API_ROOT}{API_PREFIX}/ingestas"
BASE_MEDICIONES    = f"{API_ROOT}{API_PREFIX}/mediciones"
BASE_CARACTERISTICAS = f"{API_ROOT}{API_PREFIX}/caracteristicas"
BASE_DESPLIEGUES   = f"{API_ROOT}{API_PREFIX}/despliegues"
BASE_ASSETS        = f"{API_ROOT}{API_PREFIX}/assets"
BASE_MOTORES       = f"{API_ROOT}{API_PREFIX}/motores"

# Parámetros generales
EXPECTED_SEC = 900      # intervalo esperado (15 min)
PAGE_SIZE    = 200

# ===========================
# Autenticación
# ===========================

AUTH_MODE    = "x-api-key"  # "bearer" | "basic" | "x-api-key"
BEARER_TOKEN = ""
BASIC_USER   = ""
BASIC_PASS   = ""
X_API_KEY    = "zxcvbnm"

session = requests.Session()
headers = {}
auth = None

if AUTH_MODE == "bearer" and BEARER_TOKEN:
    headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
elif AUTH_MODE == "basic" and BASIC_USER:
    auth = (BASIC_USER, BASIC_PASS)
elif AUTH_MODE == "x-api-key" and X_API_KEY:
    headers["x-api-key"] = X_API_KEY

# ===========================
# Carpetas de salida (por si las necesitas)
# ===========================

#OUTDIR = Path("eda_outputs")
#OUTDIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    # Pequeña prueba de conectividad opcional
    print("Config OK —", BASE_INGESTAS)
    print("Headers:", headers)
    test_url = f"{BASE_INGESTAS}/latest"
    r = session.get(test_url, headers=headers, auth=auth, timeout=30)
    print("Test GET", test_url, "=>", r.status_code)
    if r.status_code != 200:
        print("Respuesta:", r.text)
    else:
        print("✓ Conectividad OK")
