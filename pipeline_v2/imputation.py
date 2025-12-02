# imputation.py
import pandas as pd
import logging
from typing import Dict, Tuple

# Importaciones de conf (asumimos que todas están ahí)
from conf import (
    VIBRATION_VARS, PHYSICAL_VARS, 
    CATEGORICAL_VARS, ACCUMULATIVE_VARS, 
    METRICS_MAP, IMPUTATION_LIMIT_N
)

# Creamos un mapeo inverso para trabajar con los nombres limpios (snake_case)
INV_METRICS_MAP = {v: k for k, v in METRICS_MAP.items()}

def impute_by_group(df_ancho: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica imputación específica por grupo de variables al DataFrame Ancho (Uniforme).
    
    @param df_ancho: DataFrame en Formato Ancho y remuestreado (con NaNs).
    @return: DataFrame limpio con imputaciones.
    """
    df_imputado = df_ancho.copy()
    
    # 1. Identificar las columnas por tipo (usando nombres limpios)
    def get_clean_vars(raw_vars):
        return [METRICS_MAP[k] for k in raw_vars if k in METRICS_MAP]

    vib_phys_vars = get_clean_vars(VIBRATION_VARS + PHYSICAL_VARS)
    cat_acc_vars = get_clean_vars(CATEGORICAL_VARS + ACCUMULATIVE_VARS)
    
    # Aseguramos que solo trabajamos con columnas que existen en el DF
    vib_phys_vars = [c for c in vib_phys_vars if c in df_imputado.columns]
    cat_acc_vars = [c for c in cat_acc_vars if c in df_imputado.columns]

    # 2. Agregar bandera de imputación antes de imputar
    # Creamos una columna temporal para marcar dónde había NaN antes de la imputación.
    # Esta bandera captura gaps pequeños y outliers (si ya se reemplazaron por NaN).
    imputation_mask = df_imputado.isna().copy()

    # 3. Imputación de Continuas (Vibración y Físicas)
    if vib_phys_vars:
        logging.info("-> Imputando Continuas (Vibración/Físicas) con Interpolación Lineal...")
        # 🚨 CORRECCIÓN: Quitamos el bucle de pd.to_numeric. La columna ya debe ser float
        # debido a la corrección en 'limpiar_por_variable_deteccion'.
        df_imputado[vib_phys_vars] = df_imputado[vib_phys_vars].interpolate(
            method='linear', 
            limit=IMPUTATION_LIMIT_N,
            limit_direction='both'
        )

   # 4. Imputación de Categóricas y Acumulativas
    if cat_acc_vars:
        logging.info("-> Imputando Categóricas/Acumulativas con Forward Fill (ffill)...")
        
        # 🚨 CORRECCIÓN: Quitamos el bucle de pd.to_numeric aquí también, 
        # ya que la conversión a float se hizo en la fase de detección.
        
        df_imputado[cat_acc_vars] = df_imputado[cat_acc_vars].ffill(
            limit=IMPUTATION_LIMIT_N
        ).bfill(
            limit=IMPUTATION_LIMIT_N
        )
        # NOTA: Después de la imputación (ffill/bfill), es posible que desees convertir
        # las categóricas y acumulativas a INT de nuevo si no tienen NaNs, pero eso
        # es un paso posterior a la imputación. Por ahora, déjalas como float.

    # 5. Generar la bandera final
    # Una celda fue imputada si originalmente era NaN Y ahora tiene un valor.
    # Una celda fue imputada si originalmente era NaN Y ahora tiene un valor.
    is_imputed_flags = imputation_mask & (~df_imputado.isna())
    
    # Renombrar las columnas de la máscara para evitar conflictos y prepararlas para el melt en main.py
    is_imputed_flags.columns = [f"{col}_is_imputed" for col in is_imputed_flags.columns]
    
    # Unir las banderas al DataFrame imputado (se unirán por índice 'ts_utc')
    df_imputado = pd.concat([df_imputado, is_imputed_flags], axis=1)

    # 🚨 NOTA: Los NaNs restantes en df_imputado SÍ son GAPS GRANDES (>= 2 * 15min)
    
    return df_imputado