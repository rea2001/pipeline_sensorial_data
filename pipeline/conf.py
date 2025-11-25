# =====================================================
# 1) Definición de grupos de variables
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
    "Number of Starts Between Measurements": {0, 1},
}
