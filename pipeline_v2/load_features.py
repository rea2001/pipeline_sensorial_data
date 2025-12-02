import math
from typing import Tuple
from config_api import BASE_CARACTERISTICAS
import pandas as pd
import requests  # por si necesitas usarlo directamente
from config_api import API_ROOT, API_PREFIX, session, headers, auth


def post_with_bulk(
    df_feats: pd.DataFrame,
    batch_size: int = 500,
    timeout: int = 60,
) -> Tuple[int, int]:
    """
    Envía las características generadas (df_feats) al endpoint /caracteristicas/bulk.

    Parámetros:
      - df_feats: DataFrame con columnas:
          despliegue_id, ts_utc, variable, caracteristica,
          valor, ventana_s, indicador_calidad
      - batch_size: tamaño del lote para cada POST
      - timeout: tiempo máximo de espera por request (segundos)

    Retorna:
      (total_inserted, total_skipped)
    """
    if df_feats.empty:
        print("⚠ df_feats está vacío, no hay características para enviar.")
        return 0, 0

    required_cols = {
        "despliegue_id",
        "ts_utc",
        "variable",
        "caracteristica",
        "valor",
        "ventana_s",
        "indicador_calidad",
    }
    missing = required_cols - set(df_feats.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df_feats: {missing}")

    url = f"{BASE_CARACTERISTICAS}/bulk"
    total_rows = len(df_feats)

    total_inserted = 0
    total_skipped = 0

    print(f"\n>>> Enviando {total_rows} características en lotes de {batch_size}...")

    # Recorremos el DataFrame en chunks
    for start in range(0, total_rows, batch_size):
        end = start + batch_size
        chunk = df_feats.iloc[start:end]

        items = []
        for _, row in chunk.iterrows():
            ts_utc = row["ts_utc"]

            # Aseguramos que ts_utc sea serializable (ISO 8601)
            if pd.isna(ts_utc):
                # Si no hay ts_utc, no tiene sentido insertarlo
                continue

            if hasattr(ts_utc, "isoformat"):
                ts_str = ts_utc.isoformat()
            else:
                # Si por alguna razón no es datetime, intentamos convertir
                ts_str = pd.to_datetime(ts_utc).isoformat()

            valor = row["valor"]
            if pd.isna(valor):
                valor_json = None
            else:
                # Pydantic / FastAPI lo interpretan como Decimal si es numérico
                valor_json = float(valor)

            item = {
                "despliegue_id": int(row["despliegue_id"]),
                "ts_utc": ts_str,
                "variable": str(row["variable"]),
                "caracteristica": str(row["caracteristica"]),
                "valor": valor_json,
                "ventana_s": int(row["ventana_s"]),
                "indicador_calidad": int(row["indicador_calidad"]),
            }
            items.append(item)

        if not items:
            continue

        payload = {"items": items}

        print(f"POST {url}  (filas {start}–{end-1})  items={len(items)}")

        try:
            resp = session.post(
                url,
                json=payload,
                headers=headers,
                auth=auth,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            print(f"❌ Timeout al enviar lote {start}–{end-1}")
            # puedes decidir si quieres romper aquí o seguir:
            continue
        except Exception as e:
            print(f"❌ Error inesperado en lote {start}–{end-1}: {e}")
            continue

        if resp.status_code != 201:
            print(f"❌ Error HTTP {resp.status_code} en lote {start}–{end-1}: {resp.text}")
            continue

        try:
            data = resp.json()
        except ValueError:
            print("❌ No se pudo parsear la respuesta JSON del servidor.")
            continue

        inserted = data.get("inserted", 0)
        skipped = data.get("skipped", 0)

        total_inserted += inserted
        total_skipped += skipped

        print(f"   ✓ inserted={inserted}, skipped={skipped}")

    print(f"\n>>> RESUMEN ENVÍO CARACTERÍSTICAS")
    print(f"   Total filas en df_feats: {total_rows}")
    print(f"   Total insertadas en BD: {total_inserted}")
    print(f"   Total saltadas (skipped): {total_skipped}")

    return total_inserted, total_skipped
