import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """
    Detecta y reemplaza outliers usando el método del Rango Intercuartílico (IQR).
    
    Los valores atípicos son reemplazados por NaN y luego se interpolan.
    Este método es robusto para la mayoría de series de tiempo.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Reemplazar valores fuera de los límites por NaN
    is_outlier = (series < lower_bound) | (series > upper_bound)
    series_cleaned = series.mask(is_outlier)
    
    if is_outlier.any():
        logging.info(f"    -> Detectados {is_outlier.sum()} outliers en la serie.")
        
    return series_cleaned

def handle_missing_values(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """
    Rellena los valores faltantes (NaN) en el DataFrame.
    
    1. Forward Fill (ffill): Rellena con el último valor válido (bueno para el estado de la máquina).
    2. Interpolación Lineal: Interpola linealmente los gaps cortos restantes.
    """
    # 1. Forward Fill (rellenar con el último valor observado)
    df_filled = df.ffill()
    
    # 2. Interpolación Lineal para gaps cortos (útil después del remuestreo)
    df_interpolated = df_filled.interpolate(method='linear', limit=limit)
    
    return df_interpolated

def apply_smoothing(df: pd.DataFrame, columns: List[str], window_size: int = 3) -> pd.DataFrame:
    """
    Aplica un filtro de media móvil simple para suavizar el ruido de alta frecuencia.
    """
    for col in columns:
        if col in df.columns:
            df[f'{col}_SMOOTH'] = df[col].rolling(window=window_size, center=True).mean()
        else:
            logging.warning(f"La columna {col} no se encontró para suavizar.")
            
    return df