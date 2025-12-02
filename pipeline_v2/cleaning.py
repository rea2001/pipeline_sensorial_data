import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from conf import MAX_JUMP_THRESHOLD, STUCK_VARIANCE_THRESHOLD,STUCK_WINDOW_N, NOISE_WINDOW_N, NOISE_STD_MULTIPLIER
# Se asume que MAX_JUMP_THRESHOLD, STUCK_WINDOW_N, STUCK_VARIANCE_THRESHOLD, etc., 
# están importados desde conf.py

# 🚨 Importaciones necesarias (asumimos que conf.py tiene METRICS_MAP, RESAMPLE_FREQUENCY, etc.)
from conf import RESAMPLE_FREQUENCY, METRICS_MAP, CATEGORICAL_VARS, VIBRATION_VARS, PHYSICAL_VARS,ACCUMULATIVE_VARS, CATEGORICAL_DOMAINS

# -----------------------------------------------------------
# 1. PIVOTEO Y MANEJO DE SINCRONIZACIÓN (Formato Largo -> Ancho)
# -----------------------------------------------------------

# Se asume que RESAMPLE_FREQUENCY (ej. '15min'), METRICS_MAP, y ACCUMULATIVE_VARS 
# están accesibles dentro de esta función o se pasan como argumentos.

def _pivotar_y_mapear(df_quality: pd.DataFrame, metrics_map: dict, 
                      resample_freq: str, accumulative_vars: list) -> pd.DataFrame:
    """
    Sincroniza el tiempo, pre-agrega contadores y transforma el DF de largo a ancho.
    """
    df_proc = df_quality.copy()

    # 1. 🚨 SINCRONIZACIÓN TEMPORAL (Redondeo)
    # Redondea el ts_utc a la frecuencia de remuestreo (ej. 19:01:01 -> 19:00:00)
    df_proc['ts_utc_rounded'] = df_proc['ts_utc'].dt.floor(resample_freq)

    # 2. 🚨 PRE-AGREGACIÓN PARA MANEJO DE JITTER Y MONOTONÍA
    # Si varias lecturas caen en el mismo cubo de tiempo (debido al redondeo):
    
    # a) Identificar qué función de agregación usar por variable
    def get_agg_function(series: pd.Series):
        var_name = series.name # Nombre de la variable
        if var_name in accumulative_vars:
            # Usar MAX para Contadores (garantiza la monotonicidad y el valor más alto)
            return 'max'
        else:
            # Usar el primer valor, la media, o el último para otras variables
            # Usaremos la media o el último valor (depende de tu estándar), aquí usaremos mean() o last()
            # Por simplicidad y consistencia, usemos 'mean()' para continuas
            return 'mean' 

    # b) Agrupar por el nuevo timestamp y la variable, aplicando la función de agregación
    df_agg = df_proc.groupby(['ts_utc_rounded', 'variable'])['valor'].agg(get_agg_function).reset_index()

    # 3. PIVOTEO
    # Convertir el DataFrame de largo a ancho (cada variable es una columna)
    df_ancho = df_agg.pivot(index='ts_utc_rounded', columns='variable', values='valor')

    # 4. MAPEO Y LIMPIEZA DE COLUMNAS
    # Aplicar el mapeo de nombres
    df_ancho.columns = [metrics_map.get(col, col) for col in df_ancho.columns]
    
    # Renombrar el índice (timestamp_limpio)
    df_ancho.index.name = 'ts_utc' 
    
    return df_ancho


# -----------------------------------------------------------
# 2. REMUESTREO Y RELLENO DE GAPS (Uniformidad temporal)
# -----------------------------------------------------------

def _remuestrear_y_rellenar(df_ancho: pd.DataFrame, freq: str, interpolation_limit: int) -> pd.DataFrame:
    """
    Remuestrea a la frecuencia uniforme, interpola gaps pequeños y preserva gaps grandes.
    """
    logging.info(f"-> Remuestreando a frecuencia uniforme de {freq}...")
    
    # 1. Remuestreo base: crea un índice de tiempo uniforme (con NaNs en los huecos)
    df_resampled = df_ancho.resample(freq).asfreq()
    
    # 2. Aplicar interpolación/relleno solo si el límite es mayor a 0
    if interpolation_limit > 0:
        logging.info(f"-> Aplicando ffill/bfill con límite de {interpolation_limit} intervalos.")
        # Usamos .ffill() seguido de .bfill() con el mismo límite para interpolación simétrica
        df_resampled = df_resampled.ffill(limit=interpolation_limit).bfill(limit=interpolation_limit)
    else:
        logging.info("-> Interpolation_limit es 0. Se mantienen los NaNs en los huecos temporales.")
    
    # Flag de Missing (para gaps grandes)
    # NOTA: En este punto, los NaNs representan gaps grandes O NaNs originales
    df_resampled['is_missing_general'] = df_resampled.isna().any(axis=1)

    return df_resampled

# Se asume que los flags_cols (incluyendo los nuevos) están definidos.

def assign_final_quality_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna el código de calidad final (quality_code) siguiendo la jerarquía de severidad.
    """
    df_temp = df.copy()

    # 1. Definición de Condiciones (en orden de severidad descendente)
    conditions = [
        # Físicos y Monotónicos (Máxima Prioridad)
        df_temp['is_invalid_physical'],   # Código 5
        df_temp['is_invalid_monotonic'],  # Código 6

        # Comportamiento (Alta Prioridad)
        df_temp['is_excessive_jump'],     # Código 8
        df_temp['is_stuck_value'],        # Código 7

        # Categóricos y Extremos (Prioridad Media)
        df_temp['is_invalid_category'],   # Código 4
        df_temp['is_outlier'],            # Código 3
        df_temp['is_high'],               # (Si lo usas como flag para Código 3)

        # Temporales y Faltantes
        df_temp['is_gap'],                # Código 2
        df_temp['is_missing'],            # Código 1
        
        # Ruido (Baja Prioridad - Estructural, pero menos destructivo que un spike)
        df_temp['is_excessive_noise'],    # Código 10
        
        # Errores no clasificados
        df_temp['is_error']               # Código 9 (Asumiendo que tienes un flag para errores de procesamiento)
    ]

    # 2. Definición de Valores de Retorno (Códigos)
    choices = [5, 6, 8, 7, 4, 3, 3, 2, 1, 10, 9]

    # 3. Aplicación de la jerarquía (El primero True en 'conditions' gana)
    df_temp['quality_code'] = np.select(
        conditions,
        choices,
        default=0  # 0: OK
    )

    return df_temp


# cleaning.py (Modificación)
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


# Umbral de Tolerancia para Flotantes:
# Ignora errores de precisión donde el valor es extremadamente cercano (ej. 280.7 - 280.7 = -1e-15)
# Si el delta es menor a este valor, lo tratamos como una caída real (Código 6).
FLOAT_TOLERANCE = -1e-6 # Cualquier cambio que sea menor a -0.000001 se marca como error.


def mark_accumulative_integrity(df: pd.DataFrame, accumulative_vars: list) -> pd.DataFrame:
    """
    Detecta la caída de contadores (Código 6: Monotonicidad).
    Se aplica por variable para evitar comparar contadores diferentes.
    """
    df_proc = df.copy()
    df_proc['is_invalid_monotonic'] = False
    
    # 1. ORDENAR CRÍTICAMENTE
    # Asegúrate de ordenar el DF por tiempo, que es la clave de la detección secuencial.
    # Asumimos que la columna de tiempo es 'ts_utc' o 'ts_utc_rounded' si ya lo has creado.
    df_proc = df_proc.sort_values(by=['ts_utc', 'variable']) # Usar 'ts_utc' original si está antes de pivotar

    # 2. APLICAR COMPARACIÓN POR GRUPO
    # Esta es la forma más robusta de calcular el diferencial (diff) dentro de cada variable.
    
    # Crea una Serie booleana donde True = el valor actual es menor que el anterior
    is_negative_change_series = df_proc.groupby('variable')['valor'].transform(
        lambda x: x.diff() < FLOAT_TOLERANCE
    )

    # 3. FILTRAR SOLO LAS VARIABLES ACUMULATIVAS
    # Solo aplicamos la detección a las variables que están en la lista de contadores
    is_accumulative = df_proc['variable'].isin(accumulative_vars)
    
    # 4. ASIGNAR EL FLAG
    # El flag se marca solo si la variable es acumulativa Y el cambio fue negativo
    df_proc.loc[is_accumulative & is_negative_change_series, 'is_invalid_monotonic'] = True
    
    return df_proc



def mark_sequential_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca las anomalías que requieren análisis de comportamiento a lo largo del tiempo:
    Salto Excesivo (8), Valor Constante (7) y Ruido Excesivo (10).
    
    @param df: DataFrame en formato largo con la columna 'valor' ya en float.
    @return: DataFrame con las nuevas columnas de flags.
    """
    df_proc = df.copy()

    # Inicializar los nuevos flags a False
    df_proc['is_stuck_value'] = False
    df_proc['is_excessive_jump'] = False
    df_proc['is_excessive_noise'] = False
    
    # Lista de variables continuas que requieren esta detección
    continuous_vars = list(MAX_JUMP_THRESHOLD.keys())
    
    # ----------------------------------------------------------------------
    # A. Función de Detección Aplicada por Grupo (por variable)
    # ----------------------------------------------------------------------
    
    def apply_sequential_checks(series: pd.Series, var_name: str) -> pd.DataFrame:
        """Aplica la detección secuencial a una única serie de una variable."""
        
        # DataFrame temporal para almacenar los flags de esta variable
        flags = pd.DataFrame(index=series.index)
        
        # 1. Detección de Salto Excesivo (Código 8)
        # -----------------------------------------------------------
        # Calcula el cambio absoluto (Delta) entre el valor actual y el anterior.
        delta_v = series.diff().abs()
        
        # Obtiene el umbral específico para esta variable
        jump_threshold = MAX_JUMP_THRESHOLD.get(var_name, np.inf) 
        
        # Marca si el salto excede el umbral físico
        flags['is_excessive_jump'] = delta_v > jump_threshold

        # 2. Detección de Valor Constante (Stuck Value) (Código 7)
        # -----------------------------------------------------------
        # Calcula la desviación estándar en una ventana deslizante de N periodos
        # La ventana debe ser centrada o ajustada si se usa para el futuro (aquí usamos 'left').
        
        # Nota: Usamos la varianza o STD, si es cercana a cero.
        # min_periods=STUCK_WINDOW_N asegura que solo se calcula cuando hay suficientes datos.
        rolling_std = series.rolling(window=STUCK_WINDOW_N, min_periods=STUCK_WINDOW_N).std()
        
        # Marca si la STD es casi cero durante toda la ventana (y no es NaN/Missing)
        # Se usa una máscara para aplicar el flag a toda la ventana una vez que la condición se cumple.
        is_stuck = rolling_std < STUCK_VARIANCE_THRESHOLD
        
        # Se propaga la marca 'stuck' a lo largo de la ventana para una detección más robusta
        flags['is_stuck_value'] = is_stuck.rolling(window=STUCK_WINDOW_N).max().fillna(False).astype(bool)

        # 3. Detección de Ruido Excesivo (Código 10)
        # -----------------------------------------------------------
        # NOTA: Esta requiere un umbral estadístico (ej., 3xSTD histórica) que aún no tenemos
        # calculado, pero la lógica de la ventana es similar al 'stuck value'.
        # Por simplicidad inicial, usaremos un umbral arbitrario muy alto (DEBE SER REEMPLAZADO)
        
        rolling_noise_std = series.rolling(window=NOISE_WINDOW_N, min_periods=NOISE_WINDOW_N).std()
        
        # EJEMPLO ARBITRARIO: si la STD local es más de 10x la STD media global 
        # (Esto debe venir de los stats calculados en la Fase 1)
        # Reemplaza 'GLOBAL_STD_FOR_VAR' con la estadística que calcules.
        GLOBAL_STD_FOR_VAR = 0.5 # Valor temporal de ejemplo
        noise_threshold = GLOBAL_STD_FOR_VAR * NOISE_STD_MULTIPLIER
        
        flags['is_excessive_noise'] = rolling_noise_std > noise_threshold
        
        return flags
    
    # ----------------------------------------------------------------------
    # B. Aplicar la función de detección a todas las variables continuas
    # ----------------------------------------------------------------------
    
    for var in continuous_vars:
        # 1. Filtrar solo los datos de la variable actual
        df_var = df_proc[df_proc['variable'] == var]
        
        # 2. Aplicar los chequeos secuenciales
        # Nota: La función apply_sequential_checks necesita el nombre de la variable
        results = apply_sequential_checks(df_var['valor'], var)
        
        # 3. Mapear los resultados de los flags de vuelta al DataFrame principal
        # Las marcas se asignan solo a las filas que corresponden a esa variable
        df_proc.loc[df_var.index, 'is_stuck_value'] = results['is_stuck_value']
        df_proc.loc[df_var.index, 'is_excessive_jump'] = results['is_excessive_jump']
        df_proc.loc[df_var.index, 'is_excessive_noise'] = results['is_excessive_noise']

    return df_proc


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


def _clean_outliers_and_prepare(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    1. Marca outliers.
    2. Reemplaza outliers extremos (is_outlier=True) con NaN para futura imputación.
    """
    # 1. Marcar outliers y high (usa la lógica de mark_vibration_outliers)
    df_out = mark_vibration_outliers(df, stats)
    
    # 2. Reemplazar outliers extremos (3*IQR) por NaN para que sean imputados
    # Aplicamos esto solo a las filas que fueron marcadas como 'is_outlier'
    logging.info("-> Reemplazando outliers extremos (is_outlier=True) por NaN para imputación.")
    df_out.loc[df_out["is_outlier"], "valor"] = pd.NA # Usar pd.NA para tipo compatible con NaNs

    return df_out


def limpiar_por_variable_deteccion(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline de DETECCIÓN y Marcado de Calidad.
    - Reemplaza outliers extremos por NaN.
    - NO hace imputación ni remuestreo.
    """
    df_proc = df.copy()

    # 🚨 CORRECCIÓN CLAVE: Convertir 'valor' a float. Los valores categóricos
    # que no puedan ser convertidos a número (como "ON", "OFF", si los hubiera)
    # se harán NaN, lo cual debe ser manejado por las banderas subsiguientes.
    df_proc["valor"] = pd.to_numeric(df_proc["valor"], errors="coerce")

    # Aseguramos orden temporal (requerido para acumulativas)
    if "ts_utc" in df_proc.columns:
        df_proc = df_proc.sort_values("ts_utc")
        
    # 1) Missing (Ahora incluye los NaNs originales + NaNs por conversión fallida)
    df_proc = mark_missing(df_proc)

    # 2) Valores físicamente imposibles
    df_proc = mark_invalid_physical(df_proc) # ⚠️ Nota: Esta función ya asume que 'valor' es numérico.

    # 3) Estadísticos para vibración
    stats_vib = compute_vibration_stats(df_proc) # ⚠️ Nota: Esta función ya asume que 'valor' es numérico.

    # 4) Outliers, valores altos Y REEMPLAZO DE OUTLIERS por NaN
    df_proc = _clean_outliers_and_prepare(df_proc, stats_vib)

    # 5) Categóricas fuera de dominio
    # ⚠️ Esta función también debe manejar la columna 'valor' que ya es float (NaN si era texto).
    df_proc = mark_categorical_invalid(df_proc)

    # 6) Integridad de contadores
    df_proc = mark_accumulative_integrity(df_proc,ACCUMULATIVE_VARS)

    # ... (Aseguramos que flags existan) ...

    # 7) Quality code
    df_proc["quality_code"] = df_proc.apply(compute_quality_code, axis=1)

    # 🚨 NOTA: df_proc contiene los outliers reemplazados por NaN y el quality_code
    return df_proc, stats_vib