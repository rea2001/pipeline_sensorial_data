import time
from typing import Optional, Iterable, Tuple
import pandas as pd
import requests
from config_api import session, headers, auth, BASE_MEDICIONES

# Timeout: (connect_timeout, read_timeout)
TIMEOUT = (5, 120)
MAX_RETRIES = 5


def post_with_retry(payload: dict) -> Tuple[bool, Optional[int]]:
    """
    Envía una medición al endpoint /mediciones con reintentos.
    
    Retorna:
      (ok, status_code)
    """
    url = f"{BASE_MEDICIONES}/"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(
                url,
                json=payload,
                headers=headers,
                auth=auth,
                timeout=TIMEOUT,
            )

            # Caso éxito normal
            if r.status_code in (200, 201):
                return True, r.status_code

            # Caso conflicto por duplicado: lo tratamos como "ok lógico"
            if r.status_code == 409:
                print(f"⚠ Medición duplicada (409). Se omite pero se considera OK lógico.")
                return True, r.status_code

            # Otros errores HTTP
            print(f"⚠ Error HTTP {r.status_code}: {r.text}")
        except requests.exceptions.ReadTimeout:
            print(f"⏳ ReadTimeout en intento {attempt}")
        except requests.exceptions.ConnectionError as e:
            print(f"💥 Error de conexión en intento {attempt}: {e}")

        # Si llegó aquí, reintentamos
        sleep_time = 2 * attempt
        print(f"⏳ Reintentando en {sleep_time} segundos...")
        time.sleep(sleep_time)

    # Si se agotaron los reintentos:
    print("❌ Falló definitivamente después de varios intentos.")
    return False, None


def guardar_mediciones(
    df_limpio: pd.DataFrame,
    quality_filter: Optional[Iterable[int]] = None,
) -> Tuple[int, int, int]:
    """
    Envía mediciones limpias a la API /api/mediciones.

    Parámetros:
      df_limpio:
        DataFrame con, al menos:
          - despliegue_id
          - ts_utc (datetime)
          - variable (str)
          - valor (num o NaN)
          - quality_code (int)
      quality_filter:
        - None  → se guardan TODAS las filas.
        - [0]   → solo filas con quality_code == 0.
        - [0,3] → solo filas con quality_code en ese conjunto, etc.

    Retorna:
      (insertadas_ok, duplicadas_o_skipped_ok, fallidas)
    """
    if df_limpio.empty:
        print("⚠ df_limpio está vacío. No hay nada que guardar.")
        return 0, 0, 0

    # ------------------------------
    # Aplicar filtro de calidad
    # ------------------------------
    if quality_filter is None:
        df_a_guardar = df_limpio.copy()
        print(f"→ Guardando TODAS las {len(df_a_guardar)} mediciones procesadas")
    else:
        df_a_guardar = df_limpio[df_limpio["quality_code"].isin(quality_filter)]
        if df_a_guardar.empty:
            print("No hay mediciones que cumplan el filtro de quality_code.")
            return 0, 0, 0
        print(
            f"→ Guardando {len(df_a_guardar)} mediciones con quality_code en {list(quality_filter)} "
            f"de un total de {len(df_limpio)}."
        )

    insertadas_ok = 0
    duplicadas_o_skipped = 0
    fallidas = 0

    total = len(df_a_guardar)
    print(f"\n=== INICIO DE CARGA A /api/mediciones ===\n")
    print(f"Total de filas a enviar: {total}\n")

    for idx, row in df_a_guardar.reset_index(drop=True).iterrows():
        # Construir payload para la API
        try:
            despliegue_id = int(row["despliegue_id"])
        except KeyError:
            raise KeyError("El DataFrame debe tener la columna 'despliegue_id'.")
        except ValueError:
            raise ValueError(f"Valor inválido de despliegue_id en fila {idx}: {row['despliegue_id']}")

        ts_utc = row["ts_utc"]
        if pd.isna(ts_utc):
            # Si no hay timestamp, no tiene sentido enviarla
            print(f"⚠ Fila {idx}: ts_utc es NaN. Se omite.")
            fallidas += 1
            continue

        if not hasattr(ts_utc, "isoformat"):
            # Aseguramos que sea datetime
            ts_utc = pd.to_datetime(ts_utc)

        valor = row.get("valor", None)
        if pd.isna(valor):
            valor_json = None
        else:
            valor_json = float(valor)

        quality_code = int(row.get("quality_code", 0))

        payload = {
            "despliegue_id": despliegue_id,
            "ts_utc": ts_utc.isoformat(),
            "variable": str(row["variable"]),
            "valor": valor_json,
            "indicador_calidad": quality_code,
        }

        ok, status = post_with_retry(payload)

        if ok:
            if status == 409:
                duplicadas_o_skipped += 1
            else:
                insertadas_ok += 1
        else:
            fallidas += 1
            print(f"❌ Error al enviar medición idx={idx} (ts_utc={ts_utc}, variable={row['variable']})")

        # Log simple de progreso
        if (idx + 1) % 500 == 0:
            print(f"   → Progreso: {idx + 1}/{total} filas procesadas...")

    print("\n=== RESUMEN CARGA MEDICIONES ===")
    print(f"Insertadas OK.............: {insertadas_ok}")
    print(f"Duplicadas/omitidas.......: {duplicadas_o_skipped}")
    print(f"Fallidas..................: {fallidas}")
    print("================================\n")

    return insertadas_ok, duplicadas_o_skipped, fallidas
