import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_time_domain_features(df: pd.DataFrame, window_size: int = 12) -> pd.DataFrame:
    """
    Calcula features estadísticos sobre ventanas deslizantes para capturar tendencias.
    
    @param df: DataFrame de entrada con datos limpios y remuestreados.
    @param window_size: Número de puntos de datos para la ventana móvil (ej. 12 puntos = 1 hora si resample_freq es '5T').
    @return: DataFrame con las nuevas columnas de features.
    """
    df_features = df.copy()
    
    # Columnas de vibración (usamos las que tienen 'Vibration' y 'Acc_RMS')
    vibration_cols = [col for col in df.columns if 'vibration' in col or 'acc_rms' in col]
    
    logging.info(f"    -> Calculando features de Time-Domain para {len(vibration_cols)} columnas...")

    for col in vibration_cols:
        # Calcular la media móvil (tendencia de largo plazo)
        df_features[f'{col}_MEAN_W{window_size}'] = df_features[col].rolling(window=window_size).mean()
        
        # Calcular la desviación estándar móvil (indicador de inestabilidad o fluctuación)
        df_features[f'{col}_STD_W{window_size}'] = df_features[col].rolling(window=window_size).std()
        
        # Calcular e l valor máximo móvil (indicador de eventos pico)
        df_features[f'{col}_MAX_W{window_size}'] = df_features[col].rolling(window=window_size).max()

    return df_features

def calculate_vibration_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera ratios de vibración para diagnosticar el tipo de falla (misalignment, unbalance).
    Estos ratios son independientes de la carga, lo que los hace muy valiosos.
    """
    df_features = df.copy()
    
    logging.info("    -> Generando Ratios de Diagnóstico de Vibración...")

    # Asumiendo que las columnas Overall_Vibration o Acc_RMS ya están disponibles.
    # Si tienes Acc_RMS_Radial y Acc_RMS_Axial:
    if 'acc_rms_radial' in df.columns and 'acc_rms_axial' in df.columns:
        # Ratio Ax/Rad: Un valor alto indica Desalineación (Misalignment)
        df_features['Ratio_Axial_Radial'] = df['acc_rms_axial'] / df['acc_rms_radial']
        
    # Un índice de vibración global:
    vibration_cols = [col for col in df.columns if 'vibration' in col or 'acc_rms' in col]
    if vibration_cols:
        df_features['Vib_Energy_Total'] = df[vibration_cols].mean(axis=1)
        
    return df_features

def create_operational_features(df: pd.DataFrame, temp_col: str = 'skin_temp') -> pd.DataFrame:
    """
    Crea features basados en la condición operacional, como tasas de cambio.
    """
    df_features = df.copy()
    
    logging.info("    -> Creando Features Operacionales (Tasas de Cambio, etc.)...")

    # Tasa de cambio (Delta): Útil para la temperatura. Un cambio rápido indica un problema repentino.
    df_features[f'{temp_col}_DELTA'] = df_features[temp_col].diff()
    
    # Integración del tiempo de funcionamiento: 
    # Usar el tiempo total de funcionamiento para medir la "edad" del sensor
    if 'total_run_time' in df.columns:
        # Escalado simple para el modelo
        df_features['RUL_Proxy'] = df_features['total_run_time'] / df_features['total_run_time'].max()
        
    # Se podría añadir un feature para el Estado ON/OFF si tuvieras amperaje o caudal
    # Ejemplo: df_features['Is_ON'] = (df_features['Output_Power'] > 0.5).astype(int)

    return df_features

def run_feature_engineering(df_cleaned: pd.DataFrame, window_size: int = 12) -> pd.DataFrame:
    """
    Orquesta todas las funciones de ingeniería de características.
    """
    logging.info("== INICIANDO INGENIERÍA DE CARACTERÍSTICAS ==")
    
    df_features = calculate_time_domain_features(df_cleaned, window_size=window_size)
    df_features = calculate_vibration_ratios(df_features)
    df_features = create_operational_features(df_features)
    
    # Eliminar filas con NaN generadas por las ventanas móviles al inicio del dataset
    # Esto es CRÍTICO después de cualquier operación con .rolling() o .diff()
    df_features.dropna(inplace=True)
    
    logging.info(f"== FEATURES CREADOS. Total de columnas: {len(df_features.columns)} ==")
    return df_features