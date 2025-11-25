
from despliegue import cargar_datos_despliegue
from diagnostic_temporal import preparar_estructura_temporal, agregar_flags_temporales
from cleaning import limpiar_por_variable

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
        print(f"{k}: {v}")

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

    pausar_y_continuar("\nPresione ENTER para agregar FLAGS temporales al dataframe")

    # ---- Agregar flags temporales al DataFrame principal ----
    df_with_flags = agregar_flags_temporales(df_sorted, df_gaps)

    print("\n=== DATAFRAME ORDENADO CON FLAGS TEMPORALES ===\n")
    print(df_with_flags[["ts_utc", "variable", "valor", "is_gap", "is_small_delta"]].head(10))

    #print("Variables encontradas:", df["variable"].unique())

    pausar_y_continuar("\nPresione ENTER para iniciar la LIMPIEZA por variable")

    #dataframe con banderas temporales
    df_limpio, stats_vib = limpiar_por_variable(df_with_flags)

    print("=== Stats de vibración ===")
    print(stats_vib)

    print("=== MEDICIONES CON INDICADOR DE CALIDAD")

    print(df_limpio[["ts_utc", "variable", "valor",  "is_missing","is_invalid_physical", "is_high", "is_outlier", "is_invalid_category", "is_invalid_monotonic","quality_code"]].head(10))

    print(df_limpio[df_limpio["quality_code"] == 2].head())

    print(df_limpio[df_limpio["quality_code"] == 3][["ts_utc", "variable", "valor","is_invalid_category", "is_gap", "quality_code"]].head())


if __name__ == "__main__":
    main()
