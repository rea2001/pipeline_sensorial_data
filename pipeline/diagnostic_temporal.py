import pandas as pd
from typing import Tuple, Dict, List

# Parámetros por defecto
EXPECTED_SEC = 900      # 15 minutos
GAP_FACTOR   = 1.5      # Tolerancia para variaciones o jitter del sensor
SMALL_DELTA_FACTOR = 0.5  # La mitad del intervalo esperado


# --------- Duplicados por clave lógica en raw.ingestas ---------

KEY_COLS_DEFAULT = ["asset_codigo", "motor_codigo", "ts_utc", "variable"]
VALUE_COL_DEFAULT = "valor"


def limpiar_y_marcar_duplicados_ingestas(
    df: pd.DataFrame,
    key_cols: List[str] = KEY_COLS_DEFAULT,
    value_col: str = VALUE_COL_DEFAULT,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Detecta duplicados en la tabla raw.ingestas (formato largo) usando una clave lógica:
      key_cols = [asset_codigo, motor_codigo, ts_utc, variable].

    - Elimina duplicados exactos (misma clave + mismo valor).
    - Caso especial NaN + valor para la misma clave:
        -> conserva la fila con valor
        -> descarta las filas con NaN.
    - Duplicados conflictivos (misma clave, valores distintos no nulos):
        -> NO se eliminan
        -> se marcan con is_duplicate_key = True.

    Retorna:
      df_out  : DataFrame sin duplicados exactos ni NaN sobrantes.
      resumen : dict con conteos básicos (útil para reportes de calidad).
    """
    if df.empty:
        return df.copy(), {
            "num_exact_rows_dropped": 0,
            "num_nan_dup_dropped": 0,
            "num_conflictive_rows": 0,
            "num_keys_with_conflict": 0,
        }

    df = df.copy()

    # Asegurar que exista la columna de valor
    if value_col not in df.columns:
        raise ValueError(f"El DataFrame no tiene la columna de valor '{value_col}'.")

    # 1) Estadísticas por clave lógica
    stats = (
        df.groupby(key_cols, dropna=False)[value_col]
          .agg(
              count_rows="size",
              n_notnull=lambda s: s.notna().sum(),
              n_unique_nonnull=lambda s: s.dropna().nunique()
          )
          .reset_index()
    )

    # Hacer join para que cada fila tenga las estadísticas de su clave
    df = df.merge(stats, on=key_cols, how="left")

    # 2) Máscaras de grupos
    mask_dup_group      = df["count_rows"] > 1
    mask_conflict_group = mask_dup_group & (df["n_unique_nonnull"] > 1)

    # 3) Flag para duplicados conflictivos (valores distintos no nulos)
    df["is_duplicate_key"] = False
    df.loc[mask_conflict_group, "is_duplicate_key"] = True

    # 4) Duplicados exactos (misma clave + mismo valor)
    #    Esto también captura el caso de todos NaN con misma clave, o múltiples
    #    filas exactamente iguales en la lógica (clave + valor).
    mask_exact_dup = df.duplicated(subset=key_cols + [value_col], keep="first")

    # 5) Caso especial: grupos con NaN + UN solo valor no nulo
    #    -> no son conflicto
    #    -> se descartan los NaN y se conserva la fila con valor.
    mask_group_one_nonnull = mask_dup_group & (df["n_notnull"] == 1)
    mask_nan_value         = df[value_col].isna()
    # filas NaN en claves donde solo existe un valor no nulo
    mask_nan_dup = mask_group_one_nonnull & mask_nan_value

    # 6) Filas a eliminar:
    #    - duplicados exactos (copias extra)
    #    - NaN redundantes cuando hay un solo valor bueno en la clave
    rows_to_drop = mask_exact_dup | mask_nan_dup

    num_exact_rows_dropped = int(mask_exact_dup.sum())
    num_nan_dup_dropped    = int(mask_nan_dup.sum())

    # 7) Construir df_out limpio
    df_out = df.loc[~rows_to_drop].copy().reset_index(drop=True)

    # Limpieza de columnas auxiliares de stats
    df_out.drop(columns=["count_rows", "n_notnull", "n_unique_nonnull"], inplace=True)

    # 8) Resumen
    num_conflictive_rows = int(df_out["is_duplicate_key"].sum())
    num_keys_with_conflict = int(
        stats.loc[stats["n_unique_nonnull"] > 1, key_cols].drop_duplicates().shape[0]
    )

    resumen = {
        "num_exact_rows_dropped": num_exact_rows_dropped,
        "num_nan_dup_dropped": num_nan_dup_dropped,
        "num_conflictive_rows": num_conflictive_rows,
        "num_keys_with_conflict": num_keys_with_conflict,
    }

    return df_out, resumen




def preparar_estructura_temporal(
    df: pd.DataFrame,
    expected_sec: int = EXPECTED_SEC,
    gap_factor: float = GAP_FACTOR
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    - Ordena el DataFrame por ts_utc.
    - Calcula deltas entre timestamps únicos consecutivos.
    - Marca gaps grandes.
    
    Retorna:
      df_sorted: DataFrame original ordenado por ts_utc.
      df_gaps:   DataFrame con ts_utc, delta_s, is_gap.
      resumen:   dict con estadísticas básicas de los deltas y conteo de gaps.
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío, no se puede analizar la estructura temporal.")

    if "ts_utc" not in df.columns:
        raise ValueError("El DataFrame no tiene la columna 'ts_utc'.")

    # 1) Ordenar por ts_utc
    df_sorted = df.sort_values("ts_utc").reset_index(drop=True)

    # 2) Timestamps únicos (a nivel de muestra, no por variable)
    ts_unique = (
        df_sorted["ts_utc"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    # 3) Calcular deltas en segundos
    delta = ts_unique.diff().dt.total_seconds()

    df_gaps = pd.DataFrame({
        "ts_utc": ts_unique,
        "delta_s": delta
    })

    # El primer registro no tiene delta (es NaN)
    df_gaps.loc[0, "delta_s"] = None

    # 4) Definir umbral de gap
    gap_threshold = expected_sec * gap_factor
    df_gaps["is_gap"] = df_gaps["delta_s"] > gap_threshold

    # 5) Resumen estadístico sencillo
    delta_valid = df_gaps["delta_s"].dropna()

    resumen: Dict[str, float] = {}
    if not delta_valid.empty:
        resumen = {
            "expected_sec": float(expected_sec),
            "gap_threshold": float(gap_threshold),
            "count": int(delta_valid.count()),
            "min": float(delta_valid.min()),
            "max": float(delta_valid.max()),
            "mean": float(delta_valid.mean()),
            "std": float(delta_valid.std()),
            "p25": float(delta_valid.quantile(0.25)),
            "p75": float(delta_valid.quantile(0.75)),
            "num_gaps": int(df_gaps["is_gap"].sum()),
        }
    else:
        resumen = {
            "expected_sec": float(expected_sec),
            "gap_threshold": float(gap_threshold),
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p25": None,
            "p75": None,
            "num_gaps": 0,
        }

    return df_sorted, df_gaps, resumen

def agregar_flags_temporales(
    df_sorted: pd.DataFrame,
    df_gaps: pd.DataFrame,
    expected_sec: int = EXPECTED_SEC,
    small_delta_factor: float = SMALL_DELTA_FACTOR,
) -> pd.DataFrame:
    if df_sorted.empty:
        return df_sorted.copy()

    df = df_sorted.copy()
    gaps_idx = df_gaps.set_index("ts_utc")

    df["delta_s"] = df["ts_utc"].map(gaps_idx["delta_s"])
    df["is_gap"] = df["ts_utc"].map(gaps_idx["is_gap"]).fillna(False)

    small_delta_max = expected_sec * small_delta_factor

    df["is_small_delta"] = (
        df["delta_s"].notna()
        & (df["delta_s"] > 0)
        & (df["delta_s"] < small_delta_max)
    )

    return df