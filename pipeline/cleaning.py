import pandas as pd
from typing import Dict, Tuple, Optional
from conf import VIBRATION_VARS, PHYSICAL_VARS, CATEGORICAL_VARS, ACCUMULATIVE_VARS,CATEGORICAL_DOMAINS

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
    Calcula estadísticos y percentiles p50/p95/p99
    SOLO para variables de vibración (Grupo 1),
    asumiendo que df corresponde a un despliegue.
    """
    df_vib = df[df["variable"].isin(VIBRATION_VARS)].copy()
    df_vib["valor"] = pd.to_numeric(df_vib["valor"], errors="coerce")
    df_vib = df_vib.dropna(subset=["valor"])

    if df_vib.empty:
        return pd.DataFrame(columns=["variable", "count", "mean", "std",
                                     "min", "p50", "p95", "p99", "max"])

    stats = df_vib.groupby("variable")["valor"].agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
    )

    p50 = df_vib.groupby("variable")["valor"].quantile(0.50)
    p95 = df_vib.groupby("variable")["valor"].quantile(0.95)
    p99 = df_vib.groupby("variable")["valor"].quantile(0.99)

    stats["p50"] = p50
    stats["p95"] = p95
    stats["p99"] = p99

    stats = stats[["count", "mean", "std", "min", "p50", "p95", "p99", "max"]]
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
    Marca valores altos y extremos en variables de vibración
    usando p95 y p99 por variable.
    """
    df_out = df.copy()
    df_out["is_high"] = False
    df_out["is_outlier"] = False

    if stats.empty:
        return df_out

    p95_map: Dict[str, float] = dict(zip(stats["variable"], stats["p95"]))
    p99_map: Dict[str, float] = dict(zip(stats["variable"], stats["p99"]))

    mask_vib = df_out["variable"].isin(VIBRATION_VARS)
    df_vib = df_out[mask_vib].copy()
    df_vib["valor"] = pd.to_numeric(df_vib["valor"], errors="coerce")

    def _flags(row):
        var = row["variable"]
        val = row["valor"]
        if pd.isna(val):
            return False, False
        p95 = p95_map.get(var)
        p99 = p99_map.get(var)
        if p95 is None or p99 is None:
            return False, False
        is_out = val > p99
        is_high = (val > p95) and (val <= p99)
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
