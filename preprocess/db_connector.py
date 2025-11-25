import psycopg2
from psycopg2 import sql
import pandas as pd
from .config import DB_CONFIG, TABLES 
import logging
import sys
from sqlalchemy import create_engine
from .config import METRICS_MAP


# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DBConnector:
    """
    Maneja la conexión y las operaciones CRUD (Lectura/Escritura) con la BDTS.
    """

# ... (Dentro de la clase DBConnector) ...

    def _get_sqlalchemy_engine(self):
        """Crea y devuelve un motor SQLAlchemy para pd.to_sql."""
        try:
            db_uri = f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            return create_engine(db_uri)
        except Exception as e:
            logging.error(f"❌ Error al crear el motor SQLAlchemy: {e}")
            return None


    def insert_clean_data(self, df_clean: pd.DataFrame, tabla_destino: str, despliegue_id_prueba: int):
        """
        Inserta el DataFrame de datos limpios (Formato Ancho) en la 
        tabla de destino (Formato Largo) después de revertir los nombres a los originales de la BDTS.
        """
        engine = self._get_sqlalchemy_engine()
        if engine is None:
            return False

        # --- PASO 1: Crear el Mapeo Inverso ---
        # {Nombre_Limpio: Nombre_Original_BDTS}
        reverse_map = {v: k for k, v in METRICS_MAP.items()}
        
        # El DataFrame limpio puede tener columnas extra (_SMOOTH) que no tienen un mapeo inverso.
        # Solo trabajamos con las columnas originales que tienen mapeo en METRICS_MAP.
        cols_to_melt = list(reverse_map.keys())
        
        df_insert = df_clean.copy()
        
        # Filtrar solo las columnas que sabemos que existen en METRICS_MAP
        df_insert = df_insert[[col for col in cols_to_melt if col in df_insert.columns]]

        # PASO 2: Revertir los Nombres de Columna a los originales de la BDTS
        logging.info("-> Revertiendo nombres de columna a los originales de la BDTS...")
        df_insert.rename(columns=reverse_map, inplace=True)
        
        # PASO 3: Pivot Inverso (Melt)
        logging.info("-> Transformando el DataFrame Limpio a formato Largo (Melt)...")
        df_insert.reset_index(inplace=True, names=['ts_utc']) # Mover el índice de tiempo
        
        # El corazón de la transformación Ancho -> Largo
        df_melted = df_insert.melt(
            # ¡IMPORTANTE! Las columnas de la BDTS (ej. 'Acceleration RMS (Axial)') ahora serán los valores de 'variable'
            id_vars=['ts_utc'], 
            var_name='variable', 
            value_name='valor'
        ).dropna(subset=['valor'])

        # PASO 4: Agregar los campos requeridos y el ID de prueba
        df_melted['despliegue_id'] = despliegue_id_prueba
        df_melted['ts_local'] = df_melted['ts_utc'] 
        df_melted['archivo_origen'] = 'PreProSens_Limpieza_Basica'

        # 5. Escritura usando pd.to_sql
        schema_name = tabla_destino.split('.')[0]
        table_name = tabla_destino.split('.')[1]
        
        try:
            df_melted.to_sql(
                name=table_name,
                con=engine,
                schema=schema_name,
                if_exists='append',
                index=False
            )
            logging.info(f"✅ Escritura exitosa: {len(df_melted)} filas insertadas en {tabla_destino} (Formato Largo).")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error durante la inserción masiva en {tabla_destino}: {e}")
            logging.error("ERROR CRÍTICO: El fallo probablemente se debe a que un nombre original (ej. 'Acceleration RMS (Axial)') no existe exactamente en la tabla 'variables'.")
            return False
        
    def __init__(self, db_config=DB_CONFIG):
        self.db_config = db_config
        logging.info("Inicializado el conector de base de datos.")

    def _get_connection(self):
        """
        Intenta establecer y devolver una conexión. Maneja errores comunes.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            logging.info("Conexión a la BDTS establecida con éxito.")
            # La conexión de psycopg2 ya está configurada para autoclose si se usa 'with'
            return conn
        except psycopg2.OperationalError as e:
            logging.error(f"❌ Error al conectar a la base de datos (OperationalError): {e}")
            logging.error("Asegúrate de que la BD está corriendo y los parámetros de config.py son correctos.")
            return None
        except Exception as e:
            logging.error(f"❌ Ocurrió un error inesperado durante la conexión: {e}")
            return None


    def fetch_raw_data(self, asset_codigo: str, ts_inicio: str = None, ts_fin: str = None) -> pd.DataFrame:
        """
        Lee datos crudos de la tabla 'raw.ingestas' y devuelve un DataFrame pivoteado.
        
        Resuelve el problema de pandas/psycopg2 usando .as_string().
        """
        conn = self._get_connection()
        if conn is None:
            return pd.DataFrame() # Devuelve un DataFrame vacío en caso de fallo de conexión

        # --- CONSTRUCCIÓN DE LA CONSULTA SQL SEGURA (SOLUCIÓN A LOS ERRORES) ---
        
        # 1. Definir los componentes del nombre de la tabla (usando la clave 'raw' en minúsculas)
        schema_name = TABLES["raw"].split('.')[0]  # 'raw'
        table_name = TABLES["raw"].split('.')[1]   # 'ingestas'
        
        # 2. Construir el identificador de tabla seguro: "schema"."table"
        table_identifier = sql.SQL('{}.{}').format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name)
        )
        
        # 3. Construir la consulta con placeholders SQL ({}) y de psycopg2 (%s)
        query_template = """
            SELECT ts_utc, variable, valor
            FROM {}
            WHERE asset_codigo = %s 
            {}
            {}
            ORDER BY ts_utc;
        """
        
        # 4. Formatear la query_template con los objetos SQL (tabla y condiciones)
        final_query = sql.SQL(query_template).format(
            table_identifier,
            sql.SQL("AND ts_utc >= %s") if ts_inicio else sql.SQL(""),
            sql.SQL("AND ts_utc < %s") if ts_fin else sql.SQL("")
        )
        
        # 5. Convertir a string simple ejecutable y preparar parámetros
        params = [asset_codigo]
        if ts_inicio:
            params.append(ts_inicio)
        if ts_fin:
            params.append(ts_fin)
            
        try:
            # Usar el cursor para obtener la representación string de la consulta final
            with conn.cursor() as cur:
                executable_query_string = final_query.as_string(conn)
            
            # pd.read_sql_query ahora recibe la cadena de texto y los parámetros por separado
            df = pd.read_sql_query(executable_query_string, conn, params=params)
            
            logging.info(f"Datos crudos extraídos: {len(df)} filas antes del pivoteo.")
            
            # --- PIVOTEO A FORMATO ANCHO ---
            df_ancho = df.pivot(index='ts_utc', columns='variable', values='valor')
            df_ancho.index = pd.to_datetime(df_ancho.index, utc=True)
            return df_ancho
            
        except Exception as e:
            logging.error(f"❌ Error durante la consulta y pivoteo de datos: {e}")
            # Importante: devolver un DataFrame vacío en caso de fallo
            return pd.DataFrame() 
            
        finally:
            if conn:
                conn.close()
                

    def insert_clean_data(self, df_clean: pd.DataFrame, tabla_destino: str, despliegue_id_prueba: int):
        """
        Inserta el DataFrame de datos limpios (Formato Ancho) en la 
        tabla de destino (Formato Largo) después de revertir los nombres a los originales de la BDTS.
        """
        engine = self._get_sqlalchemy_engine()
        if engine is None:
            return False

        # --- PASO 1: Preparación de la Reversión y Filtrado ---
        # {Nombre_Limpio: Nombre_Original_BDTS}
        reverse_map = {v: k for k, v in METRICS_MAP.items()}
        
        # OBTENEMOS la lista de los Nombres Limpios que deberían ser revertidos
        columnas_limpias_originales = list(METRICS_MAP.values())
        
        df_insert = df_clean.copy()

        # Filtrar df_insert para incluir SÓLO las 22 métricas originales limpias
        # Esto evita que columnas de features o smoothing que no están en METRICS_MAP causen problemas.
        columnas_existentes = [col for col in columnas_limpias_originales if col in df_insert.columns]
        
        # 🚨 Asegúrate de que las columnas a renombrar SÍ existan, o fallará.
        if not columnas_existentes:
            logging.error("❌ No se encontraron columnas limpias originales en el DataFrame. Revisa METRICS_MAP.")
            return False

        df_insert = df_insert[columnas_existentes]


        # PASO 2: Revertir los Nombres de Columna a los originales de la BDTS
        logging.info("-> Revertiendo nombres de columna a los originales de la BDTS...")
        df_insert.rename(columns=reverse_map, inplace=True)
        
        
        # PASO 3: Pivot Inverso (Melt)
        logging.info("-> Transformando el DataFrame Limpio a formato Largo (Melt)...")
        # El índice de tiempo (ts_utc) debe ser una columna para el melt
        df_insert.reset_index(inplace=True, names=['ts_utc']) 
        
        # Los id_vars son las columnas de metadatos, el resto se 'derrite'
        id_cols = ['ts_utc']
        
        df_melted = df_insert.melt(
            id_vars=id_cols, 
            var_name='variable', 
            value_name='valor'
        ).dropna(subset=['valor'])

        # PASO 4: Agregar los campos requeridos y el ID de prueba
        df_melted['despliegue_id'] = despliegue_id_prueba
        df_melted['ts_utc'] = df_melted['ts_utc'] 
        df_melted['archivo_origen'] = 'PreProSens_Limpieza_Basica'

        # 5. Escritura usando pd.to_sql
        schema_name = tabla_destino.split('.')[0]
        table_name = tabla_destino.split('.')[1]
        
        try:
            df_melted.to_sql(
                name=table_name,
                con=engine,
                schema=schema_name,
                if_exists='append',
                index=False
            )
            logging.info(f"✅ Escritura exitosa: {len(df_melted)} filas insertadas en {tabla_destino} (Formato Largo).")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error durante la inserción masiva en {tabla_destino}: {e}")
            logging.error("Verifique que los nombres de las columnas revertidas coincidan con la tabla 'variables'.")
            return False
    
# --- Ejemplo de Uso (Para verificar la conexión) ---
if __name__ == '__main__':
    from .config import DB_CONFIG # Si usas -m, mantén .config

    # Inicializa el conector (Esto llama al __init__)
    connector = DBConnector(db_config=DB_CONFIG)
    
    # Intenta obtener una conexión de prueba
    test_conn = connector._get_connection()
    
    if test_conn:
        print("\n✅ La prueba de conexión ha sido exitosa. Cerrando conexión de prueba.")
        test_conn.close()
    else:
        print("\n❌ La prueba de conexión falló. Revisa tus credenciales.")