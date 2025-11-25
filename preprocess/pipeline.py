import pandas as pd
import numpy as np 
from .config import METRICS_MAP, RESAMPLE_FREQUENCY, OUTLIER_MULTIPLIER, SMOOTHING_WINDOWS
from .db_connector import DBConnector
from .cleaning import handle_missing_values, handle_outliers_iqr, apply_smoothing
from sklearn.preprocessing import StandardScaler
import logging
from .feature_engineering import run_feature_engineering 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreProSens:
    """
    Orquestador del Pipeline de Preprocesamiento de Datos Sensoriales.
    """
    def __init__(self, db_config):
        self.db_connector = DBConnector(db_config)
        self.scaler = StandardScaler()
        self.metrics_map = METRICS_MAP
        self.resample_freq = RESAMPLE_FREQUENCY
        
    def _remap_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el mapeo de nombres de variables a nombres de columna limpios.
        Este paso es CRÍTICO para transformar el resultado del pivoteo.
        """
        # El DataFrame ya viene pivoteado, solo necesitamos renombrar las columnas
        df.columns.name = None # Limpia el nombre del eje de columnas
        df.rename(columns=self.metrics_map, inplace=True)
        return df
        
    def remuestrear_y_alinear(self, df_ancho: pd.DataFrame) -> pd.DataFrame:
        """
        Asegura que la serie de tiempo tenga una frecuencia uniforme y rellena con la media.
        """
        logging.info(f"-> Remuestreando a frecuencia uniforme de {self.resample_freq}...")
        
        # Usamos el promedio para la mayoría de las variables de condición (vibración, temperatura)
        df_resampled = df_ancho.resample(self.resample_freq).mean()
        
        # NOTA: Contadores como Total_Running_Time deberían usar .last()
        # Si tienes estas columnas, puedes sobrescribir el promedio con el último valor:
        # if 'Total_Running_Time' in df_ancho.columns:
        #     df_resampled['Total_Running_Time'] = df_ancho['Total_Running_Time'].resample(self.resample_freq).last()

        return df_resampled

    def limpiar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la lógica de limpieza: outliers y valores faltantes.
        """
        logging.info("-> Limpiando datos (Outliers y Gaps)...")
        df_cleaned = df.copy()
        
        # 1. Manejo de Outliers (Aplicar a todas las columnas numéricas relevantes)
        for col in df_cleaned.select_dtypes(include=np.number).columns:
            # Ignoramos contadores puros, solo limpiamos métricas de condición
            if 'Total' not in col and 'Number' not in col:
                df_cleaned[col] = handle_outliers_iqr(df_cleaned[col], multiplier=OUTLIER_MULTIPLIER)
        
        # 2. Manejo de Gaps y NaN (interpolación)
        df_cleaned = handle_missing_values(df_cleaned)
        
        # 3. Suavizado (opcional pero recomendado para vibración y temperatura)
        smooth_cols = list(SMOOTHING_WINDOWS.keys())
        df_cleaned = apply_smoothing(df_cleaned, smooth_cols, window_size=3) # Usamos 3 como ejemplo
        
        # Finalmente, eliminamos cualquier fila que tenga NaN después de la interpolación (ej. al inicio del dataset)
        return df_cleaned.dropna()


    # (LOS MÉTODOS DE FEATURE ENGINEERING Y NORMALIZACIÓN SERÁN IMPLEMENTADOS DESPUÉS)
    def ingenieria_caracteristicas(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("-> Aplicando Ingeniería de Características...")
        # Lógica para crear RMS, ratios, etc.
        WINDOW_SIZE = 12 
        df_features = run_feature_engineering(df, window_size=WINDOW_SIZE)
        logging.info(f"Features creados. Dimensiones del DF: {df_features.shape}")
        return df_features


    def normalizar_datos(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Escala las características numéricas usando el StandardScaler.
        
        @param df: DataFrame de entrada con los features ya creados.
        @param fit_scaler: Si es True, entrena el escalador con estos datos.
        """
        logging.info(f"== INICIANDO NORMALIZACIÓN (fit_scaler={fit_scaler}) ==")
        
        # CORRECCIÓN 1: Usar df en lugar de df_features
        features_to_scale = df.select_dtypes(include=np.number).columns.tolist() 
        
        # CORRECCIÓN 2: Usar df.copy() para df_scaled
        df_scaled = df.copy() 
        
        # CORRECCIÓN 3: Usar df_scaled en la selección
        data_to_scale = df_scaled[features_to_scale].values
        
        try:
            if fit_scaler:
                # Entrena y transforma (solo la primera vez)
                scaled_data = self.scaler.fit_transform(data_to_scale)
                logging.info("    -> StandardScaler ENTRENADO y aplicado.")
            else:
                # Solo transforma (para lotes nuevos)
                scaled_data = self.scaler.transform(data_to_scale)
                logging.info("    -> StandardScaler APLICADO (parámetros existentes).")

            # Reemplaza las columnas originales con los datos escalados
            df_scaled[features_to_scale] = scaled_data
            
        except Exception as e:
            logging.error(f"❌ Error durante la Normalización/Escalado: {e}")
            logging.error("Esto puede deberse a que el escalador no fue entrenado (fit_scaler=True) en la primera ejecución.")
            # CORRECCIÓN 4: Devolver el argumento original df en caso de fallo.
            return df 
            
        logging.info("== NORMALIZACIÓN COMPLETADA. ==")
        return df_scaled


    def ejecutar_pipeline(self, asset_codigo: str, tabla_destino: str, ts_inicio: str = None, ts_fin: str = None, despliegue_id_prueba: int = None):
        """
        Ejecuta el flujo para la presentación: Solo lectura, limpieza, y escritura.
        """
        logging.info(f"===== INICIANDO PIPELINE para Activo: {asset_codigo} (Solo Limpieza) =====")
        
        # PASO 1, 2, 3: Lectura, Pivoteo, Remuestreo, Alineación, y Limpieza
        # ... (Mantener la lógica de los pasos 1, 2, 3 que ya tienes) ...
        
        # Ejecutando los pasos
        df_crudo = self.db_connector.fetch_raw_data(asset_codigo, ts_inicio, ts_fin)
        if df_crudo is None or df_crudo.empty: 
            logging.warning("El DataFrame de datos crudos está vacío o falló la lectura.")
            return pd.DataFrame() 

        df_ancho = self._remap_metrics(df_crudo)
        df_alineado = self.remuestrear_y_alinear(df_ancho)
        df_limpio = self.limpiar_datos(df_alineado)
        
        logging.info(f"Fase 3: Datos Limpios. Filas: {len(df_limpio)}")
        
        # PASO 4: Escritura en la BDTS (Formato Largo)
        if not df_limpio.empty:
            if despliegue_id_prueba is None:
                logging.error("❌ ERROR: Es necesario enviar el 'despliegue_id_prueba' para la inserción en 'mediciones'.")
            else:
                self.db_connector.insert_clean_data(df_limpio, tabla_destino, despliegue_id_prueba)
        
        # Devolvemos el DF limpio (Formato Ancho) para la visualización
        return df_limpio


# --- USO DEL PIPELINE (Prueba) ---
if __name__ == '__main__':
    from .config import DB_CONFIG 

    # 1. Crear el orquestador
    processor = PreProSens(db_config=DB_CONFIG)
    
    # 2. Parámetros de Prueba
    asset_id_test = 'C2000046AE' 
    # 🚨 ¡IMPORTANTE! Reemplaza '123' con un ID válido que exista en tu tabla 'despliegue'
    DESPLIEGUE_ID_PRUEBA = 2 
    
    df_resultado_limpio = processor.ejecutar_pipeline(
        asset_codigo=asset_id_test, 
        tabla_destino='iot.mediciones',
        ts_inicio='2025-02-10 00:00:00',
        ts_fin='2025-02-10 09:00:00',
        despliegue_id_prueba=DESPLIEGUE_ID_PRUEBA # Enviamos el ID directo
    )
    
    if not df_resultado_limpio.empty:
        print("\n--- ¡ESCRITURA COMPLETADA! Resultado de Limpieza (Formato Ancho) ---")
        print(df_resultado_limpio.head())
    else:
        print("\nEl pipeline terminó sin resultados. Revisa los logs y la BDTS.") 