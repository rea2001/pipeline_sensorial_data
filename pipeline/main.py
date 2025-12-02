
from despliegue import cargar_datos_despliegue
from diagnostic_temporal import preparar_estructura_temporal, agregar_flags_temporales, resolver_duplicados_muestras
from cleaning import limpiar_por_variable, aplicar_flags_a_nan
from conf import QUALITY_CODES, flag_cols, PROCESSING_CODES
from load_quality import guardar_mediciones
from features import generar_caracteristicas_despliegue 
from imputation import imputar_dataset_ancho, dataset_ancho_a_largo_con_codigos
from load_features import  post_with_bulk
from sync_timestamp import construir_dataset_ancho_sin_imputar, assign_time_slots, collapse_by_slot_and_variable
import os


def pausar_y_continuar(mensaje="Presione ENTER para continuar..."):
    input(mensaje)

def main():
    despliegue_id = int(input("\nIngrese el ID de despliegue a procesar: "))
    print()
    df = cargar_datos_despliegue(despliegue_id)

    if df.empty:
        print("No se encontraron datos para ese despliegue.")
        return
    
    df = resolver_duplicados_muestras(df)

    print("\n=== DATOS CRUDOS DEL DESPLIEGUE ===\n")
    print(df[["ingesta_id","ts_utc", "variable", "valor", "despliegue_id"]].head(10))

    print(f"\nTOTAL DE FILAS CARGADAS: {len(df)}\n")


   # PAUSA 1
    pausar_y_continuar("Presione ENTER para iniciar el ANÁLISIS TEMPORAL...")

    df_sorted, df_gaps, resumen = preparar_estructura_temporal(df)

    print("\n=== RESUMEN DE DELTAS (Δt en segundos) ===\n")
    for k, v in resumen.items():
        print(f"{k:<25}: {v}")

    # ---- Mostrar gaps detectados (si existen) ----
    gaps = df_gaps[df_gaps["is_gap"]]
    print("\n=== GAPS DETECTADOS ===\n")
    if gaps.empty:
        print("No se detectaron gaps temporales en este despliegue.\n")
    else:
        print(gaps)

    # ---- Mostrar deltas pequeños (ruido temporal) ----
    small_delta_max = int(resumen["expected_sec"] * 0.5)  # mismo factor de (SMALL_DELTA_FACTOR)

    small = df_gaps[
        (df_gaps["delta_s"].notna())
        & (df_gaps["delta_s"] > 0)
        & (df_gaps["delta_s"] < small_delta_max)
    ]

    print(f"\n=== DELTAS PEQUEÑOS (< {small_delta_max} s) ===\n")
    if small.empty:
        print("No se detectaron deltas pequeños en este despliegue.\n")
    else:
        print(small.head())

    delta_before = df_gaps["delta_s"].dropna()

    stats_before = delta_before.describe()  # incluye count, mean, std, min, 25%, 50%, 75%, max

    #pausar_y_continuar("\nPresione ENTER para agregar FLAGS temporales al dataframe...")

    # ---- Agregar flags temporales al DataFrame principal ----
    df_with_flags = agregar_flags_temporales(df_sorted, df_gaps)

    #print("\n=== DATAFRAME ORDENADO CON FLAGS TEMPORALES ===\n")
    #print(df_with_flags[["ts_utc", "variable", "valor", "is_gap", "is_small_delta"]].head(10))

    #print("Variables encontradas:", df["variable"].unique())

    pausar_y_continuar("\nPresione ENTER para verificar calidad de datos (FASE 1)...")

    #dataframe con banderas temporales
    df_limpio, stats_vib = limpiar_por_variable(df_with_flags)

    print("\n=== Stats de vibración ===")
    print(stats_vib)

    print("\n=== MEDICIONES CON INDICADOR DE CALIDAD ===\n")

    print(
        df_limpio[
            ["ts_utc", "variable", "valor",  "is_missing",
            "is_invalid_physical", "is_high", "is_outlier", 
            "is_invalid_category", "is_invalid_monotonic","quality_code"]
            ].head()
        )
    
    pausar_y_continuar("\nPresione ENTER para verficar resumen de calidad...")

    total_rows = len(df_limpio)

    print(f"\nTOTAL DE MEDICIONES: {total_rows}\n")

    quality_counts = df_limpio["quality_code"].value_counts().sort_index()
   
    print("=== CONTEO POR INDICADOR DE CALIDAD ===\n")
    for code, count in quality_counts.items():
        label = QUALITY_CODES.get(code, "Desconocido")
        perc = (count / total_rows) * 100 if total_rows > 0 else 0
        print(f"{f'Código {code} ({label})':<25} : {count} filas ({perc:.2f} %)")
    #print(df_limpio[df_limpio["quality_code"] == 2][["ts_utc", "variable", "valor","is_invalid_category", "is_gap", "quality_code"]].head())

    # 2) Resumen por banderas lógicas
   
    print("\n=== CONTEO POR BANDERAS ==\n")
    for col in flag_cols:
        if col in df_limpio.columns:
            n = int(df_limpio[col].sum())
            perc = (n / total_rows) * 100 if total_rows > 0 else 0
            print(f"  {col:<25}: {n} filas ({perc:.2f} %)")
        else:
            print(f"  {col}: (no existe en el DataFrame)")

    codigos_disponibles = list(quality_counts.keys())
    print("\n=== VISUALIZACIÓN DETALLADA DE DATOS CON INDICADOR DE CALIDAD ASIGNADO ===\n")
    print(f"Códigos de Calidad disponibles para inspeccionar: {codigos_disponibles}")
    
    while True:
        user_input = input("\nCódigo a inspeccionar (o ENTER para salir): ")

        if not user_input:
            break        
        try:
            target_code = int(user_input)
            if target_code in codigos_disponibles:
                label = QUALITY_CODES.get(target_code, "Desconocido")
                
                print(f"\n=== DATOS CON INDICADOR DE CALIDAD {target_code} ({label}) ===\n")
                df_sample = df_limpio[df_limpio["quality_code"] == target_code][
                    ["ts_utc", "variable", "valor", "quality_code"]
                ].head(10)
                print(df_sample)
            else:
                print(f" El código '{user_input}' no es válido")
                
        except ValueError:
            # Si la conversión a int falla
            print("\nEntrada no válida. Por favor, ingrese un número entero o presione ENTER para salir.")
    
 # ============================
    # OPCIÓN: GUARDAR EN iot.mediciones
    # ============================
    resp = input(
        "\n¿DESEA GUARDAR LAS MEDICIONES CON INDICADOR DE CALIDAD EN BDTS? (s/n) "
    ).strip().lower()

    if resp == "s":
        quality_filter = None
        guardar_mediciones(
            df_limpio,
            quality_filter=quality_filter,
        )
    else:
     print("\nNo se enviaron mediciones a la base de datos")

    #FASE 2: CORRECCIÓN DE ANOMALÍAS 

    pausar_y_continuar(
        "\nPresione ENTER para construir el DATASET ANCHO (sin imputación, con rejilla de 15 min)..."
    )
    

    pausar_y_continuar("\nPresione ENTER para iniciar FASE 2 (IMPUTACIÓN)...")

    # 1) Aplicar flags severas -> NaN
    df_limpio_nan = aplicar_flags_a_nan(df_limpio)
    print("ts_utc min:", df_limpio["ts_utc"].min())
    print("ts_utc max:", df_limpio["ts_utc"].max())


    # 2) Construir dataset ancho sin imputar (lo que ya hicimos antes)
    df_wide, grid = construir_dataset_ancho_sin_imputar(
        df_limpio_nan,
        freq="15min",
        jitter_max_seconds=450,
    )
     
    if not df_wide.empty:
        idx = df_wide.index.to_series()
        delta_after = idx.diff().dt.total_seconds().dropna()
        stats_after = delta_after.describe()
    
    if not delta_before.empty and not delta_after.empty:
        print("\n=== COMPARACIÓN DE Δt ANTES vs DESPUÉS ===\n")
        print(f"{'Estadística':<20} | {'CRUDA (Antes)':>18} | {'SINCRONIZADA (Después)':>24}")
        print("-" * 70)

        for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            val_before = stats_before.get(stat, None)
            val_after = stats_after.get(stat, None)
            print(f"{stat:<20} | {str(round(val_before, 2)) if val_before is not None else 'NA':>18} | {str(round(val_after, 2)) if val_after is not None else 'NA':>24}")


    df_slots = assign_time_slots(df_limpio_nan, jitter_max_seconds=450)
    print("\nFilas originales:", len(df_limpio_nan))
    print("Filas dentro de jitter:", len(df_slots))


    # Índices de las filas que eran outlier en Fase 1
    outlier_idx = df_limpio.index[df_limpio["is_outlier"]]

    # De esos índices, ¿cuáles siguen presentes en df_slots?
    survivor_idx = df_slots.index.intersection(outlier_idx)

    print("\nOutliers Fase 1 (total):", len(outlier_idx))
    print("Outliers que sobreviven a jitter (en df_slots):", len(survivor_idx))

    # Si quieres ver algunos:
    print("\nMuestra de outliers después de assign_time_slots:")
    print(df_slots.loc[survivor_idx, ["ts_utc", "ts_slot", "variable", "valor"]].head())


    if df_wide.empty:
        print("\nNo se pudo construir dataset ancho (quizá todos los datos quedaron fuera de tolerancia de jitter).")
    else:
        print("\n=== DATASET ANCHO (MUESTRA) ===\n")
        print(df_wide.head())

        print(f"\nTotal de timestamps en la rejilla: {len(df_wide.index)}")
        print(f"Rango temporal: {df_wide.index.min()}  ->  {df_wide.index.max()}")
        

    # 3) Imputación limitada (2 slots)
    df_wide_imputed, codes_wide = imputar_dataset_ancho(
        df_wide,
        max_gap_steps=2,
    )

    # ============================
    # OPCIÓN: GUARDAR DATASET ANCHO A CSV
    # ============================
    output_dir = "exports_wide"
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"despliegue_{despliegue_id}"

    # 1) Valores imputados (wide)
    csv_valores = os.path.join(output_dir, f"{base_name}_wide_valores.csv")
    df_wide_imputed.to_csv(csv_valores, index=True)
    print(f"\n[INFO] Dataset ancho de VALORES guardado en: {csv_valores}")

    # 2) Códigos de procesamiento (wide)
    csv_codigos = os.path.join(output_dir, f"{base_name}_wide_codigos.csv")
    codes_wide.to_csv(csv_codigos, index=True)
    print(f"[INFO] Dataset ancho de CÓDIGOS guardado en: {csv_codigos}")


    col = "skin_temp"  # o la que quieras
    s = df_wide[col]

    print("\nSegmentos largos de NaN para", col)
    is_na = s.isna()
    grupo = (is_na != is_na.shift()).cumsum()

    for gid, mask_seg in is_na.groupby(grupo):
        idxs = mask_seg[mask_seg].index
        if len(idxs) > 2:
            print(f"Gap de longitud {len(idxs)} desde {idxs[0]} hasta {idxs[-1]}")
            break

    codes_col = codes_wide[col]
    print("\nProcessing codes en ese gap:")
    print(codes_col.loc[idxs].value_counts())

    # 4) Pasar a formato largo con processing_code y nombre ABB
    df_preproc_long = dataset_ancho_a_largo_con_codigos(
        df_wide_imputed,
        codes_wide,
    )

    total_preproc = len(df_preproc_long)

    print("\n=== MUESTRA DATASET PREPROCESADO (LARGO) ===\n")
    print(df_preproc_long.head(20))

    print(f"\nTOTAL DE MEDICIONES PREPROCESADAS: {total_preproc}\n")

    if "processing_code" in df_preproc_long.columns and total_preproc > 0:
            print("=== CONTEO POR INDICADOR DE PROCESAMIENTO (FASE 2) ===\n")
            proc_counts = df_preproc_long["processing_code"].value_counts().sort_index()
            for code, count in proc_counts.items():
                label = PROCESSING_CODES.get(code, "Desconocido")
                perc = (count / total_preproc) * 100
                print(f"{f'Código {code} ({label})':<30} : {count} filas ({perc:.2f} %)")
    else:
            print("No se encontró la columna 'processing_code' en df_preproc_long.")

    pausar_y_continuar("\nPresione ENTER para generar CARACTERÍSTICAS de la señal principal...")

    df_feats = generar_caracteristicas_despliegue(df_limpio)

    print("\n=== CARACTERÍSTICAS GENERADAS ===\n")
    print(df_feats.head(20))        
    print(f"\nTotal de filas de características generadas: {len(df_feats)}")
    resp = input(
        "\n¿DESEA GUARDAR FEATURE VÍA API? (s/n) "
    ).strip().lower()
    if resp == "s":
        inserted, skipped = post_with_bulk(df_feats, batch_size=500, timeout=120)
 
    else:
     print("\nNo se enviaron caracteristicas a la base de datos")

if __name__ == "__main__":
    main()
