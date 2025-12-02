import pandas as pd
from typing import Tuple, Dict, List

# Parámetros por defecto
EXPECTED_SEC = 900      # 15 minutos
GAP_FACTOR   = 1.5      # Tolerancia para variaciones o jitter del sensor
SMALL_DELTA_FACTOR = 0.5  # La mitad del intervalo esperado


KEY_COLS_DEFAULT = ["asset_codigo", "motor_codigo", "ts_utc", "variable"]
VALUE_COL_DEFAULT = "valor"

def resolver_duplicados_muestras(
    df: pd.DataFrame,
    key_cols: List[str] = KEY_COLS_DEFAULT,
    keep: str = "last"
) -> pd.DataFrame:
    """
    Elimina duplicados por clave lógica (asset, motor, ts_utc, variable).

    Estrategia:
      - Ordena primero por ts_utc (y opcionalmente por ingesta_id si existe).
      - drop_duplicates(..., keep='last') => nos quedamos con el último valor
        que asumimos es el más "correcto" o actualizado.

    Retorna:
      df_sin_dups: DataFrame sin duplicados en la clave lógica.
    """
    if df.empty:
        return df

    sort_cols = ["ts_utc"]
    if "ingesta_id" in df.columns:
        sort_cols.append("ingesta_id")

    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    df_sin_dups = df_sorted.drop_duplicates(
        subset=key_cols,
        keep=keep
    ).reset_index(drop=True)

    return df_sin_dups


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