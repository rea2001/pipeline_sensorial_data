
from despliegue import cargar_datos_despliegue
from diagnostic_temporal import preparar_estructura_temporal, agregar_flags_temporales
from cleaning import limpiar_por_variable
from conf import QUALITY_LABELS, flag_cols
from load_mediciones import guardar_mediciones


def pausar_y_continuar(mensaje="Presione ENTER para continuar..."):
    input(mensaje)

def main():
    despliegue_id = int(input("\nIngrese el ID de despliegue a procesar: "))
    print()
    df = cargar_datos_despliegue(despliegue_id)

    if df.empty:
        print("No se encontraron datos para ese despliegue.")
        return

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

    pausar_y_continuar("\nPresione ENTER para agregar FLAGS temporales al dataframe...")

    # ---- Agregar flags temporales al DataFrame principal ----
    df_with_flags = agregar_flags_temporales(df_sorted, df_gaps)

    print("\n=== DATAFRAME ORDENADO CON FLAGS TEMPORALES ===\n")
    print(df_with_flags[["ts_utc", "variable", "valor", "is_gap", "is_small_delta"]].head(10))

    #print("Variables encontradas:", df["variable"].unique())

    pausar_y_continuar("\nPresione ENTER para iniciar la LIMPIEZA por variable...")

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
            ].head(10)
        )
    
    pausar_y_continuar("\nPresione ENTER para verficar resumen de calidad...")

    total_rows = len(df_limpio)

    print(f"\nTOTAL DE MEDICIONES PROCESADAS: {total_rows}\n")

    quality_counts = df_limpio["quality_code"].value_counts().sort_index()
   
    print("=== CONTEO POR INDICADOR DE CALIDAD ===\n")
    for code, count in quality_counts.items():
        label = QUALITY_LABELS.get(code, "Desconocido")
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
    print("\n=== VISUALIZACIÓN DETALLADA DE DATOS DE CALIDAD ===\n")
    print(f"Códigos de Calidad disponibles para inspeccionar: {codigos_disponibles}")
    
    while True:
        user_input = input("\nCódigo a inspeccionar (o ENTER para salir): ")

        if not user_input:
            print("\nSaliendo de la sección de inspección de códigos de calidad.")
            break        
        try:
            target_code = int(user_input)
            if target_code in codigos_disponibles:
                label = QUALITY_LABELS.get(target_code, "Desconocido")
                
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
        "\n¿Desea guardar estas mediciones en iot.mediciones vía API? (s/n) "
    ).strip().lower()

    if resp == "s":
        # Puedes elegir qué códigos de calidad quieres guardar
        # Por ejemplo: solo OK y GAP temporal
        quality_filter = [0,1,3]

        dry = input("¿Ejecutar en modo prueba (dry-run, sin escribir)? (s/n) ").strip().lower()
        dry_run = (dry == "s")

        guardar_mediciones(
            df_limpio=df_limpio,
            quality_filter=quality_filter,
            dry_run=dry_run,
        )
    else:
        print("No se enviaron mediciones a la API.")



if __name__ == "__main__":
    main()
