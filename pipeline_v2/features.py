import numpy as np
import pandas as pd
from typing import List
from conf import MAIN_SIGNAL_VAR, FEATURE_WINDOW_S, MIN_SAMPLES_PER_WINDOW


def _rms(x: np.ndarray) -> float:
    """Calcula RMS de un vector NumPy."""
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def generar_caracteristicas_despliegue(
    df_limpio: pd.DataFrame,
    ventana_s: int = FEATURE_WINDOW_S,
    min_samples: int = MIN_SAMPLES_PER_WINDOW,
) -> pd.DataFrame:
    """
    Genera características por ventanas de tiempo para un despliegue.

    Espera un df_limpio que ya tenga:
      - despliegue_id
      - ts_utc (datetime)
      - variable
      - valor
      - quality_code
      - is_gap (opcional, pero recomendable)

    Salida: DataFrame con columnas:
      - despliegue_id
      - ts_utc            (timestamp de referencia, por defecto fin de la ventana)
      - variable          (MAIN_SIGNAL_VAR)
      - caracteristica    (nombre de feature, p.ej. 'mean', 'fft_peak_amp')
      - valor
      - ventana_s
      - indicador_calidad (máximo quality_code dentro de la ventana)
    """
    if df_limpio.empty:
        return pd.DataFrame(columns=[
            "despliegue_id", "ts_utc", "variable",
            "caracteristica", "valor", "ventana_s", "indicador_calidad"
        ])

    # Aseguramos orden temporal
    df = df_limpio.sort_values("ts_utc").copy()

    # Tomamos solo la variable principal para las features
    df_sig = df[df["variable"] == MAIN_SIGNAL_VAR].copy()
    if df_sig.empty:
        # No hay esa variable en el despliegue
        return pd.DataFrame(columns=[
            "despliegue_id", "ts_utc", "variable",
            "caracteristica", "valor", "ventana_s", "indicador_calidad"
        ])

    # Aseguramos tipo numérico
    df_sig["valor"] = pd.to_numeric(df_sig["valor"], errors="coerce")

    # Extraemos despliegue_id (asumimos homogéneo en el df)
    despliegue_id = df_sig["despliegue_id"].iloc[0]

    # Rango temporal del despliegue
    t_min = df_sig["ts_utc"].min()
    t_max = df_sig["ts_utc"].max()

    # Listado donde iremos guardando dicts con las características
    rows: List[dict] = []

    # Iteramos ventanas sin solapamiento
    ventana_inicio = t_min

    while ventana_inicio < t_max:
        ventana_fin = ventana_inicio + pd.Timedelta(seconds=ventana_s)

        # Filtramos mediciones de la variable principal dentro de la ventana
        mask_win = (df_sig["ts_utc"] >= ventana_inicio) & (df_sig["ts_utc"] < ventana_fin)
        df_win = df_sig[mask_win].copy()

        if df_win.empty:
            # Avanzamos a la siguiente ventana
            ventana_inicio = ventana_fin
            continue

        # Si tenemos columna is_gap y hay un gap dentro de la ventana, opcionalmente descartamos
        if "is_gap" in df_win.columns and df_win["is_gap"].any():
            # Podrías optar por saltar la ventana
            ventana_inicio = ventana_fin
            continue

        # Separamos valores limpios (quality_code == 0)
        df_good = df_win[df_win["quality_code"] == 0].copy()
        df_good = df_good.dropna(subset=["valor"])

        if len(df_good) < min_samples:
            # Muy pocos datos confiables, descartamos la ventana
            ventana_inicio = ventana_fin
            continue

        # Vector de valores numéricos para estadísticas y FFT
        vals = df_good["valor"].to_numpy(dtype=float)

        # -------------------------
        # Estadísticos en dominio del tiempo
        # -------------------------
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        rms_val = _rms(vals)
        count_samples = int(len(vals))

        # -------------------------
        # Espectro (FFT) en dominio de la frecuencia
        # -------------------------
        # Podemos quitar la media antes de FFT para centrado
        vals_detrended = vals - mean_val

        # FFT real (solo parte positiva)
        fft_complex = np.fft.rfft(vals_detrended)
        fft_amp = np.abs(fft_complex)

        # Evitar que la componente DC (índice 0) domine, si quieres podemos ignorarla
        if fft_amp.size > 1:
            # Ignoramos bin 0 (DC) y buscamos pico en el resto
            peak_idx_rel = int(np.argmax(fft_amp[1:]) + 1)
        else:
            peak_idx_rel = 0

        fft_peak_amp = float(fft_amp[peak_idx_rel]) if fft_amp.size > 0 else 0.0
        fft_energy_total = float(np.sum(fft_amp ** 2))

        # -------------------------
        # Indicador de calidad de la ventana
        # Máximo quality_code (todas las mediciones de la variable en la ventana)
        # -------------------------
        qc_window = int(df_win["quality_code"].max()) if "quality_code" in df_win.columns else 0

        # Timestamp de referencia de la ventana (usamos el fin de la ventana
        # o el último ts_utc disponible dentro de la ventana)
        ts_ref = df_win["ts_utc"].max()

        base_info = {
            "ts_utc": ts_ref,
            "variable": MAIN_SIGNAL_VAR,
            "ventana_s": ventana_s,
            "indicador_calidad": qc_window,
             "despliegue_id": despliegue_id,

        }

        # Agregamos filas para cada característica
        rows.extend([
            {**base_info, "caracteristica": "mean",           "valor": mean_val},
            {**base_info, "caracteristica": "std",            "valor": std_val},
            {**base_info, "caracteristica": "min",            "valor": min_val},
            {**base_info, "caracteristica": "max",            "valor": max_val},
            {**base_info, "caracteristica": "rms_window",     "valor": rms_val},
            {**base_info, "caracteristica": "count_samples",  "valor": float(count_samples)},
            {**base_info, "caracteristica": "fft_peak_amp",   "valor": fft_peak_amp},
            {**base_info, "caracteristica": "fft_peak_bin",   "valor": float(peak_idx_rel)},
            {**base_info, "caracteristica": "fft_energy_total","valor": fft_energy_total},
        ])

        # Avanzamos a la siguiente ventana
        ventana_inicio = ventana_fin

    if not rows:
        return pd.DataFrame(columns=[
            "ts_utc", "variable",
            "caracteristica", "valor", "ventana_s", "indicador_calidad", "despliegue_id", 
        ])

    df_feats = pd.DataFrame(rows)
    return df_feats
