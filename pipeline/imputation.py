# phase2_impute.py

import pandas as pd
from typing import Tuple

# Importamos grupos y diccionarios desde tu conf.py
from conf import (
    VIBRATION_VARS,
    PHYSICAL_VARS,
    CATEGORICAL_VARS,
    ACCUMULATIVE_VARS,
    PROCESSING_CODES,
    METRICS_MAP,       # ABB original -> canónico
    METRICS_MAP_REV,   # canónico -> ABB original (deberías tener: METRICS_MAP_REV = {v: k for k, v in METRICS_MAP.items()})
)

# -------------------------------------------------------------------
# Estrategia de imputación según tipo de variable
# -------------------------------------------------------------------

def _estrategia_imputacion(variable_canonica: str) -> str:
    """
    Devuelve 'linear' o 'ffill' según el grupo de la variable,
    usando los nombres originales ABB y tus listas de grupos.

    variable_canonica: por ejemplo "acc_rms_axial", "skin_temp", etc.
    """
    # Pasamos de canónico -> nombre original ABB (si existe)
    original_name = METRICS_MAP_REV.get(variable_canonica, variable_canonica)

    if original_name in CATEGORICAL_VARS:
        return "ffill"
    if original_name in ACCUMULATIVE_VARS:
        return "ffill"
    if original_name in VIBRATION_VARS:
        return "linear"
    if original_name in PHYSICAL_VARS:
        return "linear"

    # Si no sabemos qué es, usamos linear por defecto
    return "linear"


# -------------------------------------------------------------------
# Imputación limitada en una sola serie
# -------------------------------------------------------------------

def _imputar_serie_limitada(
    s: pd.Series,
    metodo: str,
    max_gap_steps: int = 2,
    code_linear: int = 1,
    code_ffill: int = 2,
    code_not_imputed: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Imputa una serie temporal (índice de tiempo) con:
      - 'linear'  -> interpolación con .interpolate(method='time')
      - 'ffill'   -> relleno hacia adelante

    SOLO se imputan secuencias de NaN con longitud <= max_gap_steps.
    Las secuencias más largas se dejan como NaN y se marcan con code_not_imputed.

    Retorna:
      s_out: serie imputada
      codes: serie de processing_code por posición
    """
    s = s.copy()
    # Inicialmente todo se considera "original"
    codes = pd.Series(0, index=s.index, dtype="int8")  # 0: original

    is_na = s.isna()
    if not is_na.any():
        # No hay NaNs, todo original
        return s, codes

    # Serie auxiliar con imputación global
    if metodo == "linear":
        # Requiere DateTimeIndex; asumimos que s.index es de tipo datetime
        s_aux = s.interpolate(method="time")
        code_imputed = code_linear
    elif metodo == "ffill":
        s_aux = s.ffill()
        code_imputed = code_ffill
    else:
        # Método desconocido: no imputamos nada
        return s, codes

    # Identificar segmentos consecutivos de NaNs
    # Truco: cuando cambia is_na (True/False), incrementa el id de grupo
    grupo = (is_na != is_na.shift()).cumsum()

    for gid, mask_seg in is_na.groupby(grupo):
        idxs = mask_seg[mask_seg].index  # índices donde is_na == True en este segmento

        if len(idxs) == 0:
            continue

        segment_len = len(idxs)

        if segment_len <= max_gap_steps:
            # Gap pequeño -> imputable
            # Copiamos de la serie auxiliar
            s.loc[idxs] = s_aux.loc[idxs]
            codes.loc[idxs] = code_imputed
        else:
            # Gap grande -> no imputamos nada, se queda NaN
            codes.loc[idxs] = code_not_imputed

    # Aseguramos que cualquier posición que siga en NaN y no tenga código
    # de imputación (1 o 2) se marque como missing_not_imputed (3)
    still_na = s.isna() & (codes == 0)
    codes.loc[still_na] = code_not_imputed

    return s, codes


# -------------------------------------------------------------------
# Imputación para TODO el dataset ancho
# -------------------------------------------------------------------

def imputar_dataset_ancho(
    df_wide: pd.DataFrame,
    max_gap_steps: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica imputación limitada a df_wide (formato ancho):

    - df_wide:
        index  = ts_slot (DatetimeIndex, ej. cada 15 min)
        cols   = variables canónicas (ej. 'acc_rms_axial', 'skin_temp', ...)

    Para cada columna:
      - decide la estrategia ('linear' o 'ffill') según el tipo de variable
      - imputa sólo gaps de longitud <= max_gap_steps
      - marca processing_code por celda:

          0 = original
          1 = imputed_linear
          2 = imputed_ffill
          3 = missing_not_imputed

    Retorna:
      df_imputed: DataFrame ancho con valores imputados
      codes_wide: DataFrame ancho del mismo shape con processing_code
    """
    if df_wide.empty:
        return df_wide.copy(), df_wide.copy()

    df = df_wide.copy()

    # Aseguramos numérico en lo posible (NaNs se preservan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Inicializamos matriz de códigos en 0 (original)
    codes = pd.DataFrame(
        0,
        index=df.index,
        columns=df.columns,
        dtype="int8",
    )

    # Imputación columna por columna
    for col in df.columns:
        s = df[col]

        # Estrategia según tipo de variable (usando nombres originales ABB)
        metodo = _estrategia_imputacion(col)

        s_out, cod_col = _imputar_serie_limitada(
            s,
            metodo=metodo,
            max_gap_steps=max_gap_steps,
            code_linear=1,
            code_ffill=2,
            code_not_imputed=3,
        )

        df[col] = s_out
        codes[col] = cod_col

    return df, codes


# -------------------------------------------------------------------
# Pasar de ancho (valores + códigos) a largo
# -------------------------------------------------------------------

def dataset_ancho_a_largo_con_codigos(
    df_imputed: pd.DataFrame,
    codes_wide: pd.DataFrame,
) -> pd.DataFrame:

    # 1) Valores (preservamos NaN):
    df_val = (
        df_imputed.stack(dropna=False)   # <--- 🔥 IMPORTANTÍSIMO
        .reset_index()
    )
    df_val.columns = ["ts_slot", "variable_canonica", "valor"]

    # 2) Codes (también preservamos 0, 1, 2, 3 en todas las posiciones)
    df_codes = (
        codes_wide.stack(dropna=False)   # <--- 🔥 también necesario
        .reset_index()
    )
    df_codes.columns = ["ts_slot", "variable_canonica", "processing_code"]

    # 3) Merge explícito por ts_slot + variable_canonica
    df_long = pd.merge(
        df_val,
        df_codes,
        on=["ts_slot", "variable_canonica"],
        how="left",
    )

    # 4) Mapear la variable ABB original
    df_long["variable"] = df_long["variable_canonica"].map(METRICS_MAP_REV)
    df_long["variable"] = df_long["variable"].fillna(df_long["variable_canonica"])

    return df_long
