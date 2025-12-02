import pandas as pd
import logging
from typing import Dict, Tuple
import os 
from despliegue import cargar_datos_despliegue
from diagnostic_temporal import preparar_estructura_temporal, agregar_flags_temporales, limpiar_duplicados_raw
from cleaning import limpiar_por_variable_deteccion, compute_quality_code, _pivotar_y_mapear, _remuestrear_y_rellenar # Detección de Calidad
from imputation import impute_by_group #
from load_metrics_quality import guardar_mediciones
from features import generar_caracteristicas_despliegue 
from load_features import post_with_bulk
from conf import QUALITY_LABELS, flag_cols, RESAMPLE_FREQUENCY, METRICS_MAP, ACCUMULATIVE_VARS 


def pausar_y_continuar(mensaje="Presione ENTER para continuar..."):
    """
    Pausa la ejecución para la revisión del usuario.
    """
    input(mensaje)


def _redondear_timestamp_y_pivotar(df_largo: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Redondea 'ts_utc' a la frecuencia de remuestreo (ej. 15min) para anclar y corregir el JITTER.
    """
    df = df_largo.copy() 
    df['ts_utc'] = df['ts_utc'].dt.round(freq)
    return df


def main():
    """Flujo principal del pipeline."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    despliegue_id = int(input("\nIngrese el ID de despliegue a procesar: "))
    print()
    df = cargar_datos_despliegue(despliegue_id)
    initial_rows = len(df)

    if df.empty:
        print("No se encontraron datos para ese despliegue.")
        return

    print(f"\n 1. INGESTA Y ESTRUCTURA TEMPORAL INICIAL (ID: {despliegue_id}) ##")

    print("\n=== DATOS CRUDOS DEL DESPLIEGUE ===\n")
    print(df[["ingesta_id","ts_utc", "variable", "valor", "despliegue_id"]].head(10))
    print(f"\nTOTAL DE FILAS CARGADAS: {initial_rows}\n")


    # --- FASE 1: DIAGNÓSTICO TEMPORAL ---
    
    pausar_y_continuar("Presione ENTER para iniciar el ANÁLISIS TEMPORAL...")

    df_sorted, df_gaps, resumen = preparar_estructura_temporal(df) 

    print("\n--- RESUMEN DE DELTAS (Δt en segundos) ---\n")
    for k, v in resumen.items():
        print(f"{k:<25}: {v}")


    # ---- Mostrar gaps y deltas pequeños ----
    gaps = df_gaps[df_gaps["is_gap"]]
    print("\n--- GAPS DETECTADOS (Saltos > 2x Frecuencia) ---\n")
    if gaps.empty:
        print("No se detectaron gaps temporales.\n")
    else:
        print(gaps.head())

    small_delta_max = int(resumen["expected_sec"] * 0.5)
    small = df_gaps[
        (df_gaps["delta_s"].notna())
        & (df_gaps["delta_s"] > 0)
        & (df_gaps["delta_s"] < small_delta_max)
    ]

    print(f"\n--- DELTAS PEQUEÑOS (Jitter, < {small_delta_max} s) ---\n")
    if small.empty:
        print("No se detectaron deltas pequeños (Jitter).\n")
    else:
        print(small.head())

    pausar_y_continuar("\n 2. DETECCIÓN DE CALIDAD (DF_AUDITORIA) sobre datos CRUDOS \n")
    df_auditoria, stats_vib = limpiar_por_variable_deteccion(df_with_flags) 
    total_quality_rows = len(df_auditoria)

    print("\n--- RESUMEN DE BANDERAS DETECTADAS ---\n")
    for col in flag_cols:
        if col in df_quality.columns:
            n = int(df_quality[col].sum())
            perc = (n / total_quality_rows) * 100 if total_quality_rows > 0 else 0
            print(f"  {col:<25}: {n} filas ({perc:.2f} %)")
        else:
            print(f"  {col}: (bandera no generada)")
            
    # 1) Conteo por Indicador de Calidad
    quality_counts = df_quality["quality_code"].value_counts().sort_index()
    
    print("\n--- CONTEO POR INDICADOR DE CALIDAD ---\n")
    for code, count in quality_counts.items():
        label = QUALITY_LABELS.get(code, "Desconocido")
        perc = (count / total_quality_rows) * 100 if total_quality_rows > 0 else 0
        print(f"{f'Código {code} ({label})':<25} : {count} filas ({perc:.2f} %)")
        
    # 2) Visualización detallada de calidad
    pausar_y_continuar("\nPresione ENTER para VER DETALLE de datos con CALIDAD NO-OK (Códigos > 0)...")
    
    codigos_disponibles = list(quality_counts[quality_counts.index > 0].index)
    if not codigos_disponibles:
        print("\nTodos los datos marcados son de Calidad OK (Código 0).")
    else:
        print(f"\nCódigos de Calidad NO-OK disponibles para inspeccionar: {codigos_disponibles}")
        while True:
            user_input = input("\nCódigo a inspeccionar (o ENTER para continuar): ")
            if not user_input:
                break
            try:
                target_code = int(user_input)
                if target_code in codigos_disponibles:
                    label = QUALITY_LABELS.get(target_code, "Desconocido")
                    
                    print(f"\n=== MUESTRA DE DATOS CON INDICADOR DE CALIDAD {target_code} ({label}) ===\n")
                    df_sample = df_quality[df_quality["quality_code"] == target_code][
                        ["ts_utc", "variable", "valor", "quality_code"] + [c for c in flag_cols if c in df_quality.columns]
                    ].head(10)
                    print(df_sample)
                else:
                    print(f" El código '{user_input}' no está disponible o no tiene registros.")
            except ValueError:
                print("\nEntrada no válida. Por favor, ingrese un número entero o presione ENTER para continuar.")
    
    pausar_y_continuar("\nPresione ENTER para agregar FLAGS temporales y CORREGIR JITTER...")

    # --- FASE 2: CORRECCIÓN DE JITTER ---
    
    # 1. Agregar flags de gap y small_delta
    df_with_flags = agregar_flags_temporales(df_sorted, df_gaps)
    
    # 2. Redondeo para corregir JITTER y Sincronizar
    df_redondeado = _redondear_timestamp_y_pivotar(df_with_flags, RESAMPLE_FREQUENCY)
    
    print("\n=== DATAFRAME DESPUÉS DEL REDONDEO DE TIMESTAMP (Muestra) ===\n")
    print(df_redondeado[["ts_utc", "variable", "valor", "is_gap", "is_small_delta"]].head(10))

    # 3. VERIFICACIÓN del impacto del redondeo
    pausar_y_continuar("\nPresione ENTER para VERIFICAR el impacto del REDONDEO...")
    _, _, resumen_redondeado = preparar_estructura_temporal(df_redondeado) 

    print("\n=== RESUMEN DE DELTAS DESPUÉS DEL REDONDEO (Comparación) ===\n")
    print(f"{'Estadística':<25} | {'CRUDA (Antes)':<25} | {'REDONDEADA (Después)':<25}")
    print("-------------------------|---------------------------|---------------------------")
    for key in ["mean", "std", "min", "max", "count", "num_gaps"]:
        val_cruda = f"{resumen.get(key, 'N/A'):.2f} s"
        val_redondeada = f"{resumen_redondeado.get(key, 'N/A'):.2f} s"
        print(f"{key:<25} | {val_cruda:<25} | {val_redondeada:<25}")
    print("-------------------------|---------------------------|---------------------------")

    
    # --- FASE 3: DETECCIÓN DE CALIDAD (DF_CALIDAD) ---
    
    pausar_y_continuar("\n 2. DETECCIÓN DE CALIDAD Y REEMPLAZO DE OUTLIERS (DF_CALIDAD) \n")

    # df_quality: DF Largo con TODAS las Banderas y Quality Code. Outliers extremos son NaN.
    df_quality, stats_vib = limpiar_por_variable_deteccion(df_redondeado) 
    total_quality_rows = len(df_quality)
    
   

    # --- FASE 4: IMPUTACIÓN Y UNIFORMIDAD TEMPORAL (DF_LIMPIO) ---

    pausar_y_continuar("\n 3. LIMPIEZA ESTRUCTURAL, REMUESTREO e IMPUTACIÓN (DF_LIMPIO) \n")


    # 1. PIVOTEO Y MAPEO (Usa df_quality, que ya tiene outliers como NaN)
    # 🚨 NOTA: METRICS_MAP debe ser accesible aquí (desde conf.py)
    df_ancho = _pivotar_y_mapear(
    df_quality, 
    METRICS_MAP, 
    RESAMPLE_FREQUENCY, 
    ACCUMULATIVE_VARS # 🚨 Se pasa la lista de contadores   
    )

    # 2. REMUESTREO (Genera un índice temporal uniforme, gaps grandes y pequeños son NaN)
    # Usamos interpolation_limit=0 para NO interpolar en esta fase, solo uniformar
    df_uniforme = _remuestrear_y_rellenar(df_ancho, RESAMPLE_FREQUENCY, interpolation_limit=0)
    
    # 3. IMPUTACIÓN POR GRUPO (Rellena NaNs causados por Outliers, Jitter y Gaps pequeños)
    df_imputado_ancho = impute_by_group(df_uniforme) 
    
    # 4. REVERSIÓN A FORMATO LARGO (DF LIMPIO FINAL)
    
    # 4a. Identificar Columnas
    control_cols = ['is_missing_general']
    
    # Columnas de Valor y Columnas de Banderas de Imputación
    value_cols = [c for c in df_imputado_ancho.columns if c not in control_cols and not c.endswith('_is_imputed')]
    imputed_flag_cols = [c for c in df_imputado_ancho.columns if c.endswith('_is_imputed')]
    
    # 4b. Melt de los Valores
    df_clean_largo_values = df_imputado_ancho.reset_index()[['ts_utc'] + control_cols + value_cols].melt(
        id_vars=['ts_utc'] + control_cols, 
        value_vars=value_cols,
        var_name='variable_limpia', 
        value_name='valor_limpio'
    )

    # 4c. Melt de las Banderas de Imputación
    # Creamos un DF con solo ts_utc y las banderas
    df_imputed_flags_long = df_imputado_ancho.reset_index()[['ts_utc'] + imputed_flag_cols].melt(
        id_vars=['ts_utc'],
        value_vars=imputed_flag_cols,
        var_name='variable_imputed_flag',
        value_name='is_imputed'
    )

    # 4d. Extraer el nombre de la variable limpia de la columna de bandera
    df_imputed_flags_long['variable_limpia'] = df_imputed_flags_long['variable_imputed_flag'].str.replace('_is_imputed', '')
    
    # 4e. Fusionar los dos DataFrames largos
    df_clean_largo = pd.merge(
        df_clean_largo_values, 
        df_imputed_flags_long[['ts_utc', 'variable_limpia', 'is_imputed']], 
        on=['ts_utc', 'variable_limpia'], 
        how='left'
    )
    
    # 4f. Limpieza final de la bandera
    df_clean_largo['is_imputed'] = df_clean_largo['is_imputed'].fillna(False).astype(bool)

    total_clean_rows = len(df_clean_largo)

    print("\n=== DATAFRAME LIMPIO FINAL (Imputado, Uniforme y con Bandera de Imputación) ===\n")
    # Filtramos la muestra para que sea representativa
    print(df_clean_largo[
        ["ts_utc", "variable_limpia", "valor_limpio", "is_imputed", "is_missing_general"]
    ].head(10))
    print(f"\nTOTAL DE FILAS DEL DATAFRAME LIMPIO (Formato Largo, Uniforme): {total_clean_rows}\n")
    
    # Resumen de Imputación
    num_imputed = df_clean_largo['is_imputed'].sum()
    perc_imputed = (num_imputed / total_clean_rows) * 100 if total_clean_rows > 0 else 0
    print(f"--- RESUMEN DE IMPUTACIÓN ---")
    print(f"Total de celdas imputadas: {num_imputed} ({perc_imputed:.2f} %)")
    
    # ... (Continúa con FASE 5: PERSISTENCIA Y FEATURE ENGINEERING) ...
    # --- FASE 5: PERSISTENCIA Y FEATURE ENGINEERING ---
    
    pausar_y_continuar("\n\n#####################################################")
    pausar_y_continuar("## 4. PERSISTENCIA DE CALIDAD Y GENERACIÓN DE FEATURES ##")

    # 1. Guardado de DF de Calidad (para DB o trazabilidad)
    try:
        output_path_quality = f"calidad_despliegue_{despliegue_id}_largo.csv"
        # Renombramos 'valor' y 'variable' para que coincida con lo esperado por la DB
        df_quality_to_save = df_quality.rename(columns={'variable': 'variable_original'})
        df_quality_to_save.to_csv(output_path_quality, index=False)
        logging.info(f"✅ DF de Calidad (sin imputación) guardado en: {output_path_quality}")
    except Exception as e:
        logging.error(f"❌ Error al intentar guardar el archivo de CALIDAD: {e}")

    # 2. Guardado de DF Limpio (para trazabilidad/Debugging)
    try:
        output_path_clean = f"limpieza_despliegue_{despliegue_id}_limpio_largo.csv"
        df_clean_largo.to_csv(output_path_clean, index=False)
        logging.info(f"✅ DF Limpio Final (con imputación) guardado en: {output_path_clean}")
    except Exception as e:
        logging.error(f"❌ Error al intentar guardar el archivo LIMPIO: {e}")
    
    # 4. Persistencia de Features (Comentado, asumiendo que ya tienes la función)
    # post_with_bulk(df_features) 

if __name__ == "__main__":
    main()