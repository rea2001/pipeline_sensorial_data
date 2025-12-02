
RESAMPLE_FREQUENCY = '15min'  # Remuestreo de 15
IMPUTATION_LIMIT_N = 2

# Variable principal para análisis espectral
MAIN_SIGNAL_VAR = "Acceleration RMS (Radial)"

# Tamaño de ventana en segundos (2 horas = 7200 s)
FEATURE_WINDOW_S = 7200
# Número mínimo de muestras "buenas" (quality_code == 0) para aceptar una ventana
MIN_SAMPLES_PER_WINDOW = 4

#---------
# --- Parámetros de Comportamiento Secuencial (Códigos 7 y 10) ---
STUCK_WINDOW_N = 8 # 2 horas a 15 min.
STUCK_VARIANCE_THRESHOLD = 1e-6 
NOISE_WINDOW_N = 4 # 1 hora a 15 min.
NOISE_STD_MULTIPLIER = 3.0
# NOTA: NOISE_STD_MULTIPLIER (3.0) aún debe ser usado para comparar contra el histórico.
# (Código 8: Salto Excesivo) ---
# Define el cambio absoluto máximo permitido en un solo intervalo de 15 minutos.

MAX_JUMP_THRESHOLD = {
    # I. VARIABLES FÍSICAS
    'Skin Temperature': 10.0,      # (C)
    'Motor Supply Frequency': 10.0,  # (Hz)
    'Speed': 1800.0,               # (rpm)
    'Output Power': 260.0,         # (kW)
    
    # II. VARIABLES DE VIBRACIÓN (RMS / mm/s)
    
    # Overall Vibration (mm/s RMS)
    "Overall Vibration": 2.0, 
    
    # Vibration (Ejes: mm/s)
    "Vibration (Tangential)": 2.0,   
    "Vibration (Radial)": 2.0,
    "Vibration (Axial)": 2.0,
    
    # Peak to Peak (P-P: G)
    "Peak to Peak (Tangential)": 5.0, 
    "Peak to Peak (Radial)": 5.0,
    "Peak to Peak (Axial)": 5.0,
    "Peak to Peak": 5.0,
    
    # Acceleration RMS (G RMS)
    "Acceleration RMS (Tangential)": 2.0,
    "Acceleration RMS (Radial)": 2.0,
    "Acceleration RMS (Axial)": 2.0,
}

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
    "Peak to Peak": "peak_to_peak",
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
    "Number of Starts Between Measurements": "num_starts_betw_meas",
}

# =====================================================
# Definición de grupos de variables
# =====================================================

# Grupo 1: continuas físicas de vibración
VIBRATION_VARS = [
    "Vibration (Tangential)",
    "Vibration (Radial)",
    "Vibration (Axial)",
    "Overall Vibration",
    "Peak to Peak (Tangential)",
    "Peak to Peak (Radial)",
    "Peak to Peak (Axial)",
    "Peak to Peak",
    "Acceleration RMS (Tangential)",
    "Acceleration RMS (Radial)",
    "Acceleration RMS (Axial)",
]

# Grupo 2: continuas físicas no relacionadas ccon vibración
PHYSICAL_VARS = [
    "Skin Temperature",       # °C
    "Motor Supply Frequency", # Hz
    "Speed",                  # rpm
    "Output Power",           # kW
]

# Grupo 3: categóricas / diagnósticas
CATEGORICAL_VARS = [
    "Bearing Condition",
    "Unbalance (BETA)",
    "Misalignment (BETA)",
    "Looseness (BETA)",
    "Number of Starts Between Measurements",
]

# Grupo 4: acumulativas / contadores
ACCUMULATIVE_VARS = [
    "Total Number Of Starts",
    "Total Running Time",
]


# =====================================================
# 2) Dominios categóricos empíricos
# =====================================================

CATEGORICAL_DOMAINS = {
    "Bearing Condition": {0, 1, 2, 3, 4},
    "Unbalance (BETA)": {0, 1},
    "Misalignment (BETA)": {0, 1},
    "Looseness (BETA)": {0, 1},
    "Number of Starts Between Measurements": {0, 1,2},
}

QUALITY_LABELS = {
        0: "OK",
        1: "Valor faltante",
        2: "Gap temporal",
        3: "Valor extremo",
        4: "Categoría inválida",
        5: "Valor imposible",
        6: "Error de integridad en contador",
        ##7: "Valor constante (Stuck)",       
        #8: "Salto excesivo (Spike)",  
        9: "Error desconocido",      
        #10: "Ruido excesivo (High Noise)"
}

flag_cols = [
        "is_missing",
        "is_invalid_physical",
        "is_high", # para feacture
        "is_outlier",
        "is_invalid_category",
        "is_invalid_monotonic",
        "is_gap",           # viene de df_with_flags / para feature
        "is_small_delta", 
        #"is_stuck_value",        # NEW: Para Código 7
        #"is_excessive_jump",     # NEW: Para Código 8
        #"is_excessive_noise",  # también temporal / para feature
]
