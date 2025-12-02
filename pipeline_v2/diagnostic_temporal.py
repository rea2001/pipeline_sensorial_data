import pandas as pd
from typing import Tuple, Dict, List

# Parámetros por defecto
EXPECTED_SEC = 900      # 15 minutos
GAP_FACTOR   = 1.5      # Tolerancia para variaciones o jitter del sensor
SMALL_DELTA_FACTOR = 0.5  # La mitad del intervalo esperado


KEY_COLS_DEFAULT = ["asset_codigo", "motor_codigo", "ts_utc", "variable"]
VALUE_COL_DEFAULT = "valor"

def limpiar_duplicados_raw(
    df: pd.DataFrame,
    key_cols: List[str] = KEY_COLS_DEFAULT,
    value_col: str = VALUE_COL_DEFAULT,
) -> pd.DataFrame:
    """
    Identifica y elimina filas duplicadas basadas en las columnas clave (key_cols).
    Asume que el DataFrame ya está ordenado o que el último duplicado es el
    registro 'válido'.
    
    Por defecto, usa: ["asset_codigo", "motor_codigo", "ts_utc", "variable"]
    Conservamos el 'last' (último) que se asume es el más reciente/válido en la base de datos.
    """
    if df.empty:
        print("DataFrame vacío, no hay duplicados que limpiar.")
        return df

    # Identificar filas duplicadas
    dups_mask = df.duplicated(subset=key_cols, keep=False)
    
    # Contar duplicados
    num_dups = dups_mask.sum()
    num_a_eliminar = dups_mask.sum() - df[dups_mask].duplicated(subset=key_cols, keep='last').sum()

    if num_dups > 0:
        print(f"✅ Se encontraron {num_dups} registros que forman parte de un duplicado.")
        print(f"🗑️ Se eliminarán {num_a_eliminar} registros duplicados (conservando el último).")
    else:
        print("✅ No se encontraron registros duplicados.")
    
    # Eliminar duplicados, manteniendo el ÚLTIMO
    df_clean = df.drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)
    
    return df_clean

def sincronizar_y_pivotar_datos(
    df: pd.DataFrame,
    freq: str = "15min", 
    time_col: str = "ts_utc",
    variable_col: str = "variable",
    value_col: str = "valor",
    key_cols: List[str] = ["asset_codigo", "motor_codigo"], # Columnas de indexación
) -> pd.DataFrame:
    """
    Redondea el timestamp a una frecuencia fija (sincronización),
    y pivota el DataFrame de formato largo (ingestas) a formato ancho (series de tiempo).
    
    Args:
        df: DataFrame de ingestas (formato largo).
        freq: Cadena de frecuencia de redondeo de pandas (ej. '15min', '30T', '1H').
        ...
        
    Retorna:
        pd.DataFrame: DataFrame sincronizado en formato ancho.
    """
    if df.empty:
        print("DataFrame vacío, retornando DataFrame vacío en formato ancho.")
        return pd.DataFrame()
    
    df = df.copy()
    
    # 1. Redondear el timestamp para sincronizar
    # Redondea ts_utc al múltiplo más cercano de la frecuencia
    df["ts_rounded"] = df[time_col].dt.round(freq=freq)
    
    print(f"🔄 Timestamps redondeados a la frecuencia: **{freq}**")

    # 2. Pivotar el DataFrame: Largo -> Ancho (automáticamente llena con NaN donde no hay dato)
    # Las columnas clave + el timestamp redondeado formarán el nuevo índice.
    # 'variable' serán las nuevas columnas.
    # 'valor' serán los datos.
    
    try:
        # Se asegura de que solo haya un valor por (key_cols, ts_rounded, variable)
        df_pivot = df.pivot_table(
            index=key_cols + ["ts_rounded"],
            columns=variable_col,
            values=value_col,
            aggfunc='first' # En caso de que queden duplicados EXACTOS, toma el primer valor.
        ).reset_index()
        
    except ValueError as e:
        # Esto ocurre si después del redondeo y el 'drop_duplicates' en el paso 1,
        # quedan múltiples valores para la misma celda (ts_rounded, variable).
        print(f"⚠️ Error al pivotar. Podría haber múltiples valores para la misma celda después del redondeo: {e}")
        # Intentar una agregación por la media si pivot_table falla
        print("Intentando agregar por la media...")
        df_pivot = df.pivot_table(
            index=key_cols + ["ts_rounded"],
            columns=variable_col,
            values=value_col,
            aggfunc='mean' # Agregamos promediando si hay conflicto
        ).reset_index()

    # 3. Renombrar la columna de tiempo y establecer el índice
    df_pivot = df_pivot.rename(columns={"ts_rounded": time_col})
    df_pivot = df_pivot.set_index(time_col).sort_index()
    
    print(f"✅ DataFrame sincronizado y en formato ancho con {len(df_pivot.columns)} variables.")
    print(f"📊 La estructura final es (Index: ts_utc, Columns: {', '.join(df[variable_col].unique()[:3])}... [otras 19 variables])")

    return df_pivot

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
