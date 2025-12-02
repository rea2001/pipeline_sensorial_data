# phase2_sync.py

import pandas as pd
from typing import Tuple, Optional
from conf import METRICS_MAP

# ==========================================================
# 1) Construir rejilla (grid) de 15 minutos
# ==========================================================

def build_time_grid(
    ts_min: pd.Timestamp,
    ts_max: pd.Timestamp,
    freq: str = "15min"
) -> pd.DatetimeIndex:
    """
    Construye una rejilla regular (grid) de timestamps
    entre ts_min y ts_max, alineando a múltiplos de 'freq'.
    """
    # floor al inicio y ceil al final
    start = ts_min.floor(freq)
    end = ts_max.ceil(freq)

    grid = pd.date_range(start=start, end=end, freq=freq)
    return grid


# ==========================================================
# 2) Asignar cada medición a un "slot" de la rejilla (jitter)
# ==========================================================

def assign_time_slots(
    df: pd.DataFrame,
    freq: str = "15min",
    jitter_max_seconds: int = 60
) -> pd.DataFrame:
    """
    Asigna cada ts_utc al "slot" más cercano de la rejilla de 'freq'.
    - Usa dt.round(freq) para proponer el slot.
    - Si la diferencia absoluta entre ts_utc y slot > jitter_max_seconds,
      se considera fuera de tolerancia (no jitter) y se descarta del
      dataset preprocesado (pero ojo: los datos originales ya quedaron
      en Fase 1, no se pierden como evidencia).
    """
    if df.empty:
        return df.copy()

    if "ts_utc" not in df.columns:
        raise ValueError("El DataFrame no tiene la columna 'ts_utc'.")

    df2 = df.copy()
    df2["ts_utc"] = pd.to_datetime(df2["ts_utc"])

    # Proponer slot redondeado
    df2["ts_slot"] = df2["ts_utc"].dt.round(freq)

    # Diferencia absoluta en segundos entre el original y el slot
    diff = (df2["ts_utc"] - df2["ts_slot"]).dt.total_seconds().abs()

    # Bandera de dentro de tolerancia de jitter
    df2["is_within_jitter"] = diff <= jitter_max_seconds

    # Nos quedamos solo con las mediciones que consideramos jitter corregible
    df_in = df2[df2["is_within_jitter"]].copy()

    return df_in


# ==========================================================
# 3) Colapsar múltiples lecturas por slot (último valor)
# ==========================================================

def collapse_by_slot_and_variable(df_slots: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa todas las mediciones que caen en el mismo:
      (despliegue_id, asset_codigo, motor_codigo, ts_slot, variable)
    quedándonos con el ÚLTIMO valor.

    Esto resuelve:
      - múltiples lecturas dentro del mismo intervalo de 15 minutos
      - evita promediar variables acumulativas (no queremos romper monotonicidad)
    """
    if df_slots.empty:
        return df_slots.copy()

    df = df_slots.copy()

    # Columnas de contexto (si existen)
    context_cols = [
        c for c in ["despliegue_id", "asset_codigo", "motor_codigo"]
        if c in df.columns
    ]

    group_cols = context_cols + ["ts_slot", "variable"]

    # Aseguramos que 'valor' sea numérico donde tenga sentido
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # Ordenamos para que "last" sea realmente el último temporalmente
    sort_cols = []
    if "ts_utc" in df.columns:
        sort_cols.append("ts_utc")
    if "ingesta_id" in df.columns:
        sort_cols.append("ingesta_id")

    if sort_cols:
        df = df.sort_values(sort_cols)

    df_collapsed = (
        df.groupby(group_cols, as_index=False)
          .agg({"valor": "last"})
    )

    return df_collapsed


# ==========================================================
# 4) Aplicar mapeo ABB -> nombre canónico para pivot
# ==========================================================

def aplicar_mapeo_canonico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columna 'variable_canonica' usando METRICS_MAP.
    Si una variable no está en el diccionario, se deja tal cual.
    """
    if df.empty:
        return df.copy()

    if "variable" not in df.columns:
        raise ValueError("Falta columna 'variable' en el DataFrame.")

    df2 = df.copy()
    df2["variable_canonica"] = df2["variable"].map(METRICS_MAP)
    
    df2["variable"] = df2["variable_canonica"].fillna(df2["variable"])


    return df2


# ==========================================================
# 5) Pivot a formato ancho → estado del activo
# ==========================================================

def pivotar_estado_activo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a formato ancho:
      index   = ts_slot
      columns = variable_canonica
      values  = valor
    """
    if df.empty:
        return df.copy()

    for col in ["ts_slot", "variable_canonica", "valor"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}' en el DataFrame.")

    df_pivot = df.pivot(
        index="ts_slot",
        columns="variable_canonica",
        values="valor"
    )

    df_pivot.sort_index(inplace=True)
    df_pivot.index.name = "ts_slot"

    return df_pivot


# ==========================================================
# 6) Remuestreo: asegurar rejilla completa 15 min
# ==========================================================

def remuestrear_rejilla_completa(
    df_pivot: pd.DataFrame,
    freq: str = "15min"
) -> pd.DataFrame:
    """
    Asegura una rejilla completa entre el primer y último ts_slot:
      - index = ts_slot
      - se crean filas adicionales con NaN donde no había datos
    """
    if df_pivot.empty:
        return df_pivot.copy()

    df = df_pivot.sort_index()

    ts_min = df.index.min()
    ts_max = df.index.max()

    grid = build_time_grid(ts_min, ts_max, freq=freq)

    df_resampled = df.reindex(grid)
    df_resampled.index.name = "ts_slot"

    return df_resampled


# ==========================================================
# 7) Función principal: construir dataset ancho SIN imputar
# ==========================================================

def construir_dataset_ancho_sin_imputar(
    df_limpio: pd.DataFrame,
    freq: str = "15min",
    jitter_max_seconds: int = 60
) -> Tuple[pd.DataFrame, Optional[pd.DatetimeIndex]]:
    """
    Fase 2 - Parte 1: Sincronización temporal sin imputación.

    Entrada:
      df_limpio (formato largo, salida de Fase 1):
        - ts_utc
        - variable (ABB original)
        - valor
        - despliegue_id, asset_codigo, motor_codigo (opcional)

    Pasos:
      1) Asignar cada ts_utc al slot más cercano de la rejilla (jitter).
      2) Filtrar mediciones fuera de tolerancia de jitter.
      3) Colapsar múltiples lecturas por (slot, variable) tomando el último valor.
      4) Mapear variable -> variable_canonica para pivot.
      5) Pivotar: estado del activo (formato ancho).
      6) Remuestrear la rejilla para asegurar un timestamp cada 15 min.

    Salida:
      df_wide: DataFrame ancho con:
        - index = ts_slot (15 min)
        - cols = variable_canonica
        - valores = valor (NaN donde faltan datos)
      grid: DatetimeIndex de la rejilla utilizada (o None si df_limpio estaba vacío)
    """
    if df_limpio.empty:
        return df_limpio.copy(), None

    # 1) Asignar slots de tiempo corrigiendo jitter
    df_slots = assign_time_slots(
        df_limpio,
        freq=freq,
        jitter_max_seconds=jitter_max_seconds
    )

    if df_slots.empty:
        # Todos quedaron fuera de tolerancia de jitter
        return pd.DataFrame(), None

    # 2) Colapsar múltiples lecturas por (slot, variable) usando último valor
    df_collapsed = collapse_by_slot_and_variable(df_slots)

    # 3) Mapear a nombres canónicos
    df_mapped = aplicar_mapeo_canonico(df_collapsed)

    # 4) Pivotar a formato ancho (estado del activo)
    df_pivot = pivotar_estado_activo(df_mapped)

    # 5) Remuestrear para asegurar rejilla completa de 15 min
    df_wide = remuestrear_rejilla_completa(df_pivot, freq=freq)

    # Grid final (índice de df_wide)
    grid = df_wide.index

    return df_wide, grid
