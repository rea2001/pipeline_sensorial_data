# --------------------------------------------------------------------------
# CONFIGURACIÓN DE CONEXIÓN A LA BASE DE DATOS (POSTGRESQL)
# --------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",         
    "database": "database",
    "user": "postgres",
    "password": "ale2001",
    "port": "5432"                  
}

# --------------------------------------------------------------------------
# CONFIGURACIÓN DE NOMBRES DE TABLAS
# --------------------------------------------------------------------------
TABLES = {
    "raw": "raw.ingestas",
    "iot": "iot.mediciones"       
}

# --------------------------------------------------------------------------
# MAPEO DE MÉTRICAS (DB LONG) A COLUMNAS (DF ANCHO)
# --------------------------------------------------------------------------
METRICS_MAP = {
    "Speed": "speed",
    "Skin Temperature": "skin_temp",
    "Overall Vibration": "overall_vibration",
    "Acceleration RMS (Axial)": "acc_rms_axial",
    "Acceleration RMS (Radial)": "acc_rms_radial",
    "Acceleration RMS (Tangential)": "acc_rms_tangential",
    "Bearing Condition": "bearing_condition",
    "Peak to Peak (Axial)": "peak_to_peak_axial",
    "Peak to Peak (Radial)": "peak_to_peak_radial",
    "Peak to Peak (Tangential)": "peak_to_peak_tangential",
    "Vibration (Axial)": "vibration_axial",
    "Vibration (Radial)": "vibration_radial",
    "Vibration (Tangential)": "vibration_tangential",
    "Misalignment (BETA)": "misalignment_beta",
    "Total Running Time": "total_run_time",
    "Total Number of Starts": "total_num_starts",
    "Unbalance (BETA)": "unbalance_beta",
    "Motor Supply Frequency": "motor_supply_freq",
    "Output Power": "output_power",
    "Looseness (BETA)": "looseness_beta",
    "Number of starts between Measurements": "num_starts_betw_meas",
}

# --------------------------------------------------------------------------
# PARÁMETROS DE PREPROCESAMIENTO
# --------------------------------------------------------------------------
RESAMPLE_FREQUENCY = '5T'  # Remuestreo a 5 minutos (5T)
OUTLIER_MULTIPLIER = 3.0   # Multiplicador IQR (3.0 es común)
WINDOW_SIZE = 12 
SMOOTHING_WINDOWS = {      # Columnas a suavizar y tamaño de ventana
    "acc_rms_axial": 3,
    "acc_rms_radial": 3,
    "acc_rms_tangential": 3,
    "skin_temp": 5  # La temperatura es más lenta, ventana mayor
}