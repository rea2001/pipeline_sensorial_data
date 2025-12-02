import pandas as pd
from typing import Dict, Tuple, Optional
from conf import VIBRATION_VARS, PHYSICAL_VARS, CATEGORICAL_VARS, ACCUMULATIVE_VARS,CATEGORICAL_DOMAINS


# phase2_flags.py

import pandas as pd
from typing import List

FLAG_AS_MISSING_COLS = [
    "is_invalid_physical",
    "is_invalid_category",
    "is_invalid_monotonic",
    "is_outlier",
    "is_missing"
    # podrías incluir aquí otras flags que quieras tratar como faltantes
]

def aplicar_flags_a_nan(df_limpio: pd.DataFrame) -> pd.DataFrame:
    if df_limpio.empty:
        return df_limpio.copy()

    df = df_limpio.copy()

    if "valor" not in df.columns:
        raise ValueError("Falta la columna 'valor' en df_limpio.")

    mask_severa = False
    for col in FLAG_AS_MISSING_COLS:
        if col in df.columns:
            mask_severa = mask_severa | df[col].astype(bool)

    df.loc[mask_severa, "valor"] = pd.NA

    return df


def classify_variable_group(var_name: str) -> str:
    """
    Devuelve el grupo de variable:
      - 'vibration'
      - 'physical'
      - 'categorical'
      - 'accumulative'
      - 'unknown'
    """
    if var_name in VIBRATION_VARS:
        return "vibration"
    if var_name in PHYSICAL_VARS:
        return "physical"
    if var_name in CATEGORICAL_VARS:
        return "categorical"
    if var_name in ACCUMULATIVE_VARS:
        return "accumulative"
    return "unknown"

# =====================================================
#  Estadísticos y percentiles para vibración
# =====================================================

def compute_vibration_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticos descriptivos e IQR para variables de vibración.

    - Aplica solo a VIBRATION_VARS.
    - Devuelve, por variable:
        count, mean, std, min, q1, q3, iqr, max
    """
    df_vib = df[df["variable"].isin(VIBRATION_VARS)].copy()
    df_vib["valor"] = pd.to_numeric(df_vib["valor"], errors="coerce")
    df_vib = df_vib.dropna(subset=["valor"])

    if df_vib.empty:
        return pd.DataFrame(columns=[
            "variable", "count", "mean",
            "min", "q1", "q3", "iqr", "max"
        ])

    # Estadísticos básicos
    stats = df_vib.groupby("variable")["valor"].agg(
        count="count",
        mean="mean",
        min="min",
        max="max",
    )

    # Cuartiles e IQR
    q1 = df_vib.groupby("variable")["valor"].quantile(0.25)
    q3 = df_vib.groupby("variable")["valor"].quantile(0.75)
    iqr = q3 - q1

    stats["q1"] = q1
    stats["q3"] = q3
    stats["iqr"] = iqr

    stats = stats[["count", "mean", "min","max", "q1", "q3", "iqr"]]
    return stats.reset_index()


# =====================================================
# 4) Marcación de anomalías por grupo
# =====================================================

def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["is_missing"] = df_out["valor"].isna()
    return df_out


def mark_invalid_physical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca valores físicamente imposibles:
      - cualquier valor numérico < 0 en continuas o acumulativas
    """
    df_out = df.copy()
    df_out["is_invalid_physical"] = False

    mask_num = df_out["variable"].isin(VIBRATION_VARS + PHYSICAL_VARS + ACCUMULATIVE_VARS)
    df_num = df_out[mask_num].copy()

    df_num["valor"] = pd.to_numeric(df_num["valor"], errors="coerce")

    invalid_idx = df_num.index[df_num["valor"] < 0]
    df_out.loc[invalid_idx, "is_invalid_physical"] = True

    return df_out


def mark_vibration_outliers(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Marca valores altos y extremos en variables de vibración usando IQR.

    Criterio por variable:
      - Se calculan Q1, Q3, IQR a partir de compute_vibration_stats().
      - Zona alta (is_high=True):
          valor < Q1 - 1.5*IQR  o  valor > Q3 + 1.5*IQR
        (siempre que no sea extremo)
      - Outlier extremo (is_outlier=True):
          valor < Q1 - 3*IQR  o  valor > Q3 + 3*IQR

    NOTA:
      - is_outlier tiene prioridad: si es outlier, no se marca como high.
      - Si IQR <= 0 o faltan Q1/Q3, no se marcan flags para esa variable.
    """
    df_out = df.copy()
    df_out["is_high"] = False
    df_out["is_outlier"] = False

    if stats.empty:
        return df_out

    # Mapas por variable
    q1_map: Dict[str, float] = dict(zip(stats["variable"], stats["q1"]))
    q3_map: Dict[str, float] = dict(zip(stats["variable"], stats["q3"]))
    iqr_map: Dict[str, float] = dict(zip(stats["variable"], stats["iqr"]))

    # Filtramos solo variables de vibración
    mask_vib = df_out["variable"].isin(VIBRATION_VARS)
    df_vib = df_out[mask_vib].copy()
    df_vib["valor"] = pd.to_numeric(df_vib["valor"], errors="coerce")

    def _flags(row):
        var = row["variable"]
        val = row["valor"]

        if pd.isna(val):
            return False, False

        q1 = q1_map.get(var)
        q3 = q3_map.get(var)
        iqr = iqr_map.get(var)

        # Si falta algo o IQR no es positivo, no marcamos
        if q1 is None or q3 is None or iqr is None or iqr <= 0:
            return False, False

        # Umbrales
        low_high = q1 - 1.5 * iqr
        high_high = q3 + 1.5 * iqr

        low_out = q1 - 3.0 * iqr
        high_out = q3 + 3.0 * iqr

        # Outlier extremo
        is_out = (val < low_out) or (val > high_out)

        # Zona alta (solo si no es extremo)
        is_high = False
        if not is_out and (val < low_high or val > high_high):
            is_high = True

        return is_high, is_out

    flags = df_vib.apply(_flags, axis=1, result_type="expand")
    flags.columns = ["is_high_tmp", "is_outlier_tmp"]

    df_out.loc[mask_vib, "is_high"] = flags["is_high_tmp"].values
    df_out.loc[mask_vib, "is_outlier"] = flags["is_outlier_tmp"].values

    return df_out

def mark_categorical_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca valores categóricos fuera de dominio.
    Maneja correctamente valores como 0, 1, 0.0, 1.0, "1", "1.0", etc.
    """
    df_out = df.copy()
    df_out["is_invalid_category"] = False

    mask_cat = df_out["variable"].isin(CATEGORICAL_VARS)
    df_cat = df_out[mask_cat].copy()

    if df_cat.empty:
        return df_out

    # Convertimos a numérico una sola vez (acepta "1", "1.0", 1, 1.0, etc.)
    df_cat["valor_num"] = pd.to_numeric(df_cat["valor"], errors="coerce")

    for var, group in df_cat.groupby("variable"):
        raw_domain = CATEGORICAL_DOMAINS.get(var)
        if raw_domain is None:
            continue

        # Normalizamos el dominio a enteros (por si vienen como strings)
        valid_vals = {int(v) for v in raw_domain}

        for idx, row in group.iterrows():
            val_num = row["valor_num"]
            if pd.isna(val_num):
                # Missing se trata aparte, no como inválido de dominio
                continue
            try:
                val_int = int(val_num)
            except Exception:
                df_out.at[idx, "is_invalid_category"] = True
                continue

            if val_int not in valid_vals:
                df_out.at[idx, "is_invalid_category"] = True

    return df_out



def mark_accumulative_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica integridad de contadores:
      - no deben decrecer a lo largo del tiempo dentro del despliegue.
    Requiere:
      - df ordenado por ts_utc
    """
    df_out = df.copy()
    df_out["is_invalid_monotonic"] = False

    mask_acc = df_out["variable"].isin(ACCUMULATIVE_VARS)
    df_acc = df_out[mask_acc].copy()

    if df_acc.empty:
        return df_out

    # Aseguramos orden temporal
    df_acc = df_acc.sort_values(["variable", "ts_utc"])
    df_acc["valor"] = pd.to_numeric(df_acc["valor"], errors="coerce")

    for var, group in df_acc.groupby("variable"):
        prev_val: Optional[float] = None
        for idx, row in group.iterrows():
            val = row["valor"]
            if pd.isna(val):
                prev_val = val
                continue
            if prev_val is not None and not pd.isna(prev_val) and val < prev_val:
                df_out.at[idx, "is_invalid_monotonic"] = True
            prev_val = val

    return df_out


# =====================================================
# 5) Indicador de calidad (quality_code)
# =====================================================

def compute_quality_code(row: pd.Series) -> int:
    """
    Aplica la jerarquía de severidad para asignar un código de calidad
    compatible con iot.indicador_calidad.
    """
    # Por si acaso no existen todas las columnas (se usan get)
    invalid_monotonic = bool(row.get("is_invalid_monotonic", False))
    invalid_physical = bool(row.get("is_invalid_physical", False))
    invalid_category = bool(row.get("is_invalid_category", False))
    is_outlier = bool(row.get("is_outlier", False))
    is_gap = bool(row.get("is_gap", False))  # viene de módulo temporal
    is_missing = bool(row.get("is_missing", False))

    # Jerarquía de más severo a menos severo
    if invalid_monotonic:
        return 6  # Error de integridad en contador
    if invalid_physical:
        return 5  # Valor físicamente imposible
    if invalid_category:
        return 4  # Categoría inválida
    if is_outlier:
        return 3  # Valor extremo (>p99)
    if is_gap:
        return 2  # Gap temporal
    if is_missing:
        return 1  # Valor faltante
    return 0      # OK


# =====================================================
# 6) Función principal de limpieza por variable
# =====================================================

def limpiar_por_variable(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline de limpieza por variable para un despliegue:

    Entrada:
      df con, al menos:
        - ts_utc
        - variable
        - valor
        - (opcional) is_gap desde el módulo de estructura temporal

    Pasos:
      - Marca valores faltantes.
      - Marca valores físicamente imposibles en continuas y acumulativas.
      - Calcula estadísticos y percentiles para variables de vibración.
      - Marca valores altos y extremos en vibración.
      - Valida dominios de variables categóricas.
      - Verifica integridad en variables acumulativas.
      - Genera el código de calidad (quality_code).

    Salida:
      - df_limpio: DataFrame con banderas + quality_code
      - stats_vib: tabla de estadísticos p50/p95/p99 para variables de vibración
    """
    df_proc = df.copy()

    # Aseguramos orden temporal (para acumulativas)
    if "ts_utc" in df_proc.columns:
        df_proc = df_proc.sort_values("ts_utc")

    # 1) Missing
    df_proc = mark_missing(df_proc)

    # 2) Valores físicamente imposibles
    df_proc = mark_invalid_physical(df_proc)

    # 3) Estadísticos para vibración
    stats_vib = compute_vibration_stats(df_proc)

    # 4) Outliers y valores altos en vibración
    df_proc = mark_vibration_outliers(df_proc, stats_vib)

    # 5) Categóricas fuera de dominio
    df_proc = mark_categorical_invalid(df_proc)

    # 6) Integridad de contadores
    df_proc = mark_accumulative_integrity(df_proc)

    # Aseguramos que columnas de flags existan (por si algún grupo no aplicó)
    for col in [
        "is_missing",
        "is_invalid_physical",
        "is_high",
        "is_outlier",
        "is_invalid_category",
        "is_invalid_monotonic",
    ]:
        if col not in df_proc.columns:
            df_proc[col] = False

    # 7) Quality code
    df_proc["quality_code"] = df_proc.apply(compute_quality_code, axis=1)

    return df_proc, stats_vib
