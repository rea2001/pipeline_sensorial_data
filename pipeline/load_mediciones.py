import pandas as pd
import datetime as dt
from typing import Iterable, Optional, Sequence
from config_api import BASE_MEDICIONES

from config_api import (
    API_ROOT,
    API_PREFIX,
    session,
    headers,
    auth,
)

def _to_iso_utc(ts) -> str:
    """
    Convierte un timestamp a ISO 8601 con timezone.
    Asume que ya viene como datetime con tzinfo=UTC o similar.
    """
    if isinstance(ts, dt.datetime):
        # Si no tiene tzinfo, asumimos UTC
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.isoformat()
    # Si viene como string, lo enviamos tal cual (idealmente no pasa)
    return str(ts)


def construir_payload_medicion(row: pd.Series) -> dict:
    """
    Construye el JSON para el POST /mediciones a partir de una fila del df_limpio.
    Requiere que el DataFrame tenga:
      - despliegue_id
      - ts_utc
      - variable
      - valor
      - quality_code (que mapeamos a indicador_calidad)
    """
    despliegue_id = int(row["despliegue_id"])
    ts_utc = _to_iso_utc(row["ts_utc"])
    variable = str(row["variable"])

    # valor puede ser NaN
    valor = row.get("valor", None)
    if pd.isna(valor):
        valor = None
    else:
        # Convertimos a float explícito para JSON
        valor = float(valor)

    quality_code = int(row.get("quality_code", 0))

    payload = {
        "despliegue_id": despliegue_id,
        "ts_utc": ts_utc,
        "variable": variable,
        "valor": valor,
        "indicador_calidad": quality_code,
    }
    return payload


def post_medicion(payload: dict) -> int:
    """
    Envía una medición individual al endpoint POST /mediciones.
    Devuelve el status_code de la respuesta.
    Lanza excepción si no es 2xx.
    """
    url = BASE_MEDICIONES
    r = session.post(url, json=payload, headers=headers, auth=auth, timeout=(5,120))

    if r.status_code not in (200, 201):
        # Log sencillo; si quieres puedes imprimir r.text
        raise RuntimeError(f"Error al crear medición: {r.status_code} - {r.text}")

    return r.status_code


def guardar_mediciones(
    df_limpio: pd.DataFrame,
    quality_filter: Optional[Sequence[int]] = None,
    dry_run: bool = False,
) -> None:
    """
    Recorre df_limpio y envía las mediciones a /mediciones.

    Parámetros:
      - df_limpio:
          DataFrame resultante de limpiar_por_variable().
          Debe tener:
            despliegue_id, ts_utc, variable, valor, quality_code
      - quality_filter:
          Secuencia de códigos de calidad a incluir.
          Por ejemplo:
             [0]        -> solo mediciones OK
             [0, 2]     -> OK + gap temporal
             None       -> todas las mediciones
      - dry_run:
          Si True, no hace POST, solo muestra cuántas se enviarían.

    Imprime un pequeño resumen al final.
    """
    if df_limpio.empty:
        print("⚠ No hay mediciones para guardar (df_limpio está vacío).")
        return

    required_cols = {"despliegue_id", "ts_utc", "variable", "valor", "quality_code"}
    missing = required_cols - set(df_limpio.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df_limpio: {missing}")

    df_to_send = df_limpio.copy()

    # Filtrar por indicador_calidad si se pasó quality_filter
    if quality_filter is not None:
        df_to_send = df_to_send[df_to_send["quality_code"].isin(quality_filter)]

    total = len(df_to_send)
    if total == 0:
        print("⚠ No hay mediciones que cumplan el filtro de quality_code.")
        return

    print(f"\n=== INICIO DE CARGA A /mediciones ===")
    print(f"Total de mediciones a procesar: {total}")
    if quality_filter is not None:
        print(f"Filtro de quality_code aplicado: {list(quality_filter)}")
    if dry_run:
        print("Modo DRY-RUN activado: no se enviarán POST reales.\n")

    ok_count = 0
    error_count = 0

    for idx, row in df_to_send.iterrows():
        payload = construir_payload_medicion(row)

        if dry_run:
            # Para debug: muestra las primeras filas
            if ok_count < 3:
                print(f"[DRY-RUN] Ejemplo de payload: {payload}")
            ok_count += 1
            continue

        try:
            status = post_medicion(payload)
            ok_count += 1
            # Si quieres feedback, puedes imprimir cada N filas:
            if ok_count % 500 == 0:
                print(f"  ...{ok_count} mediciones enviadas correctamente")
        except Exception as e:
            error_count += 1
            print(f"❌ Error al enviar medición idx={idx}: {e}")

    print("\n=== RESUMEN CARGA /mediciones ===")
    print(f"Enviadas correctamente: {ok_count}")
    print(f"Con error:             {error_count}")
    print(f"Total procesadas:      {ok_count + error_count}")
    if dry_run:
        print("*(Modo DRY-RUN: no se realizaron inserciones reales en la BD)*")
