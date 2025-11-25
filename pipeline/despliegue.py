import pandas as pd
import datetime as dt

from config_api import (
    session, headers, auth,
    BASE_DESPLIEGUES, BASE_ASSETS, BASE_MOTORES, BASE_INGESTAS
)

def get_despliegue_by_id(despliegue_id: int) -> dict:
    """
    Obtiene un despliegue desde la API usando list + filtro,
    porque de momento no hay GET /despliegues/{id}.
    Devuelve un dict con asset_id, motor_id, inicio, fin, etc.
    """
    url = BASE_DESPLIEGUES
    params = {"limit": 10}  #parametro dinámico
    r = session.get(url, params=params, headers=headers, auth=auth, timeout=30)
    if r.status_code != 200:
        print("Error al listar despliegues:", r.status_code, r.text)
        r.raise_for_status()

    items = r.json()
    for d in items:
        if d.get("despliegue_id") == despliegue_id:
            return d

    raise ValueError(f"Despliegue {despliegue_id} no encontrado en la lista")

def get_asset_codigo(asset_id: int) -> str:
    """
    Consulta la API de assets para obtener el asset_codigo a partir del asset_id.
    """
    url = f"{BASE_ASSETS}/{asset_id}"
    r = session.get(url, headers=headers, auth=auth, timeout=30)
    if r.status_code != 200:
        print("Error al consultar asset:", r.status_code, r.text)
        r.raise_for_status()
    data = r.json()
    # Ajusta 'asset_codigo' al nombre real del campo
    return data["asset_codigo"]

def get_motor_codigo(motor_id: int) -> str:
    """
    Consulta la API de motores para obtener el motor_codigo a partir del motor_id.
    """
    url = f"{BASE_MOTORES}/{motor_id}"
    r = session.get(url, headers=headers, auth=auth, timeout=30)
    if r.status_code != 200:
        print("Error al consultar motor:", r.status_code, r.text)
        r.raise_for_status()
    data = r.json()
    return data["motor_codigo"]

# ==========
# Ingestas
# ==========

def load_via_by_asset_motor(
    asset_codigo: str,
    motor_codigo: str,
    t_from: dt.datetime,
    t_to: dt.datetime
) -> pd.DataFrame:
    """
    Versión modular de tu función para traer ingestas por asset+motor+rango.
    """
    url = f"{BASE_INGESTAS}/by-asset-motor" 
    params = {
        "asset_codigo": asset_codigo,
        "motor_codigo": motor_codigo,
        "ts_from": t_from.isoformat(),
        "ts_to": t_to.isoformat(),
    }
    etiquetas = {
        "asset_codigo": "CÓDIGO DE ASSET",
        "motor_codigo": "CÓDIGO DE MOTOR",
        "ts_from": "DESDE (ts_from)",
        "ts_to": "HASTA (ts_to)",
    }
    for key, value in params.items():
        etiqueta = etiquetas.get(key, key.capitalize())
        print(f"| {etiqueta:<25}: {value}")

    #print("GET :", params)

    r = session.get(url, params=params, headers=headers, auth=auth, timeout=60)
    if r.status_code != 200:
        print("Error al traer ingestas:", r.status_code, r.text)
        r.raise_for_status()

    payload = r.json()
    df = pd.DataFrame(payload.get("items", []))

    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    if "ts_local_tz" in df.columns:
        df["ts_local_tz"] = pd.to_datetime(df["ts_local_tz"])

    return df

# ==========
# Función principal por despliegue
# ==========

def cargar_datos_despliegue(despliegue_id: int) -> pd.DataFrame:
    """
    1) Obtiene datos del despliegue (asset_id, motor_id, inicio, fin).
    2) Resuelve asset_codigo y motor_codigo vía API.
    3) Llama al endpoint de ingestas por asset+motor+rango.
    4) Devuelve un DataFrame con TODAS las variables del despliegue (formato largo).
    """

    # 1. Datos del despliegue (dict)
    d = get_despliegue_by_id(despliegue_id)

    asset_id = d["asset_id"]
    motor_id = d["motor_id"]
    inicio   = pd.to_datetime(d["inicio"])
    fin      = pd.to_datetime(d["fin"]) if d["fin"] is not None else None

    if fin is None:
        raise ValueError("El despliegue tiene fin = NULL (activo). Para este pipeline, se requiere fin definido.")

    # 2. Resolver códigos (string) a partir de IDs
    asset_codigo = get_asset_codigo(asset_id)
    motor_codigo = get_motor_codigo(motor_id)

    # 3. Traer ingestas crudas del despliegue
    df = load_via_by_asset_motor(
        asset_codigo=asset_codigo,
        motor_codigo=motor_codigo,
        t_from=inicio,
        t_to=fin
    )

    # 4. Añadir despliegue_id para trazabilidad
    if not df.empty:
        df["despliegue_id"] = despliegue_id

    return df 
    
