import os
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime, timedelta, date
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuración de parámetros
DIAS_RECIENTES = 28  # Analizar solo el último mes
MIN_FALTAS_CONSECUTIVAS = 3
UMBRAL_FALTA_ALTA = 0.65
UMBRAL_FALTA_MEDIA = 0.5
UMBRAL_RETARDO = 0.4

def conectar_db():
    mysql_url = os.getenv("MYSQL_URL")
    if not mysql_url:
        raise ValueError("La variable de entorno MYSQL_URL no está configurada")
    
    result = urlparse(mysql_url)
    return mysql.connector.connect(
        host=result.hostname,
        user=result.username,
        password=result.password,
        database=result.path[1:],
        port=result.port
    )

def obtener_datos_recientes():
    try:
        conexion = conectar_db()
        cursor = conexion.cursor(dictionary=True)  # Usar dictionary=True para obtener resultados como diccionarios
        
        # Calcular fecha límite (hoy - DIAS_RECIENTES)
        fecha_limite = (datetime.now() - timedelta(days=DIAS_RECIENTES)).strftime('%Y-%m-%d')
        
        # Obtener periodo actual
        fecha_actual = datetime.now()
        anio = fecha_actual.year
        mes = fecha_actual.month
        periodo = 1 if (mes >= 8 or mes <= 1) else 2
        
        # Obtener id_periodo actual
        query_periodo = "SELECT id_periodo FROM periodos WHERE anio = %s AND periodo = %s"
        cursor.execute(query_periodo, (anio, periodo))
        periodo_result = cursor.fetchone()
        
        if not periodo_result:
            raise Exception('No existe el periodo actual.')
        
        id_periodo = periodo_result['id_periodo']
        
        # Obtener id_contenedor del periodo actual
        cursor.execute("SELECT id_contenedor FROM contenedor WHERE id_periodo = %s", (id_periodo,))
        contenedor_result = cursor.fetchone()
        
        if not contenedor_result:
            raise Exception('No existe contenedor para el periodo actual.')
        
        id_contenedor = contenedor_result['id_contenedor']
        
        # Consulta optimizada para datos recientes
        query = """
        SELECT
            p.id_persona,
            h.id_horario,
            h.dia_horario,
            a.fecha_asistencia,
            CASE
                WHEN a.validacion_asistencia = 1 THEN 2
                WHEN r.validacion_retardo = 1 THEN 1
                WHEN f.validacion_falta = 1 THEN 0
                ELSE NULL
            END AS tipo_asistencia
        FROM horario h
        JOIN persona p ON h.id_persona = p.id_persona
        LEFT JOIN asistencia a ON a.id_horario = h.id_horario AND a.fecha_asistencia >= %s
        LEFT JOIN retardo r ON r.id_horario = h.id_horario AND r.fecha_retardo = a.fecha_asistencia
        LEFT JOIN falta f ON f.id_horario = h.id_horario AND f.fecha_falta = a.fecha_asistencia
        WHERE h.id_contenedor = %s
        ORDER BY p.id_persona, h.id_horario, a.fecha_asistencia DESC
        """
        
        cursor.execute(query, (fecha_limite, id_contenedor))
        resultados = cursor.fetchall()
        
        if not resultados:
            raise Exception('No hay datos recientes para analizar.')
        
        df = pd.DataFrame(resultados)
        return df
        
    except Exception as e:
        print(f"Error al obtener datos: {str(e)}")
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conexion' in locals():
            conexion.close()

def detectar_faltas_recientes_consecutivas(df):
    try:
        if df.empty:
            return pd.DataFrame()
            
        # Filtrar solo faltas (tipo_asistencia = 0) y convertir fechas
        df_faltas = df[df['tipo_asistencia'] == 0].copy()
        df_faltas['fecha_asistencia'] = pd.to_datetime(df_faltas['fecha_asistencia'])
        
        if df_faltas.empty:
            return pd.DataFrame()
        
        # Ordenar por profesor, horario y fecha
        df_faltas = df_faltas.sort_values(['id_persona', 'id_horario', 'fecha_asistencia'])
        
        # Calcular diferencia entre fechas consecutivas
        df_faltas['diff_dias'] = df_faltas.groupby(['id_persona', 'id_horario'])['fecha_asistencia'].diff().dt.days.fillna(0)
        
        # Identificar secuencias de faltas consecutivas (máximo 7 días entre faltas)
        df_faltas['consecutivo'] = (df_faltas['diff_dias'] > 7).cumsum()
        
        # Contar faltas consecutivas por grupo
        conteo_faltas = df_faltas.groupby(['id_persona', 'id_horario', 'consecutivo']).agg(
            num_faltas=('tipo_asistencia', 'size'),
            ultima_falta=('fecha_asistencia', 'max')
        ).reset_index()
        
        # Filtrar solo los que tienen MIN_FALTAS_CONSECUTIVAS o más
        faltas_significativas = conteo_faltas[conteo_faltas['num_faltas'] >= MIN_FALTAS_CONSECUTIVAS]
        
        return faltas_significativas[['id_persona', 'id_horario', 'num_faltas', 'ultima_falta']]
        
    except Exception as e:
        print(f"Error al detectar faltas consecutivas: {str(e)}")
        return pd.DataFrame()

def preparar_datos(df):
    try:
        if df.empty:
            raise ValueError("DataFrame vacío recibido para preparación de datos")
            
        # Limpiar y transformar datos
        df = df.dropna(subset=['tipo_asistencia'])
        df['fecha_asistencia'] = pd.to_datetime(df['fecha_asistencia'])
        df['dia_semana'] = df['fecha_asistencia'].dt.dayofweek
        df['dia_mes'] = df['fecha_asistencia'].dt.day
        
        # Calcular estadísticas recientes por profesor/horario
        stats = df.groupby(['id_persona', 'id_horario']).agg(
            total_clases=('tipo_asistencia', 'count'),
            faltas=('tipo_asistencia', lambda x: sum(x == 0)),
            retardos=('tipo_asistencia', lambda x: sum(x == 1))
        ).reset_index()
        
        stats['porc_falta'] = stats['faltas'] / stats['total_clases']
        stats['porc_retardo'] = stats['retardos'] / stats['total_clases']
        
        # Unir estadísticas al dataframe original
        df = df.merge(stats, on=['id_persona', 'id_horario'])
        
        X = df[['dia_semana', 'dia_mes', 'porc_falta', 'porc_retardo']]
        y = df['tipo_asistencia']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        print(f"Error al preparar datos: {str(e)}")
        raise

def guardar_prediccion(id_persona, id_horario, resultado):
    try:
        conexion = conectar_db()
        cursor = conexion.cursor(dictionary=True)
        
        # Obtener id_escuela
        cursor.execute("SELECT id_escuela FROM horario WHERE id_horario = %s", (id_horario,))
        escuela_result = cursor.fetchone()
        
        if not escuela_result:
            print(f"✖ No se encontró escuela para horario {id_horario}")
            return
        
        id_escuela = escuela_result['id_escuela']
        
        # Insertar predicción
        cursor.execute("""
            INSERT INTO predicciones 
            (id_persona, id_horario, id_escuela, fecha_prediccion, resultado_prediccion)
            VALUES (%s, %s, %s, %s, %s)
        """, (int(id_persona), int(id_horario), int(id_escuela), date.today(), resultado))
        
        conexion.commit()
        print(f"✔ Predicción guardada: {id_persona} - {resultado}")
        
    except Exception as e:
        print(f"Error al guardar predicción: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conexion' in locals():
            conexion.close()

def ejecutar_predicciones():
    print("Iniciando proceso de predicción...")
    
    try:
        # Limpiar predicciones anteriores
        try:
            conexion = conectar_db()
            cursor = conexion.cursor()
            cursor.execute("DELETE FROM predicciones")
            conexion.commit()
            print("✔ Predicciones anteriores borradas")
        except Exception as e:
            print(f"⚠ Error al borrar predicciones anteriores: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conexion' in locals():
                conexion.close()
        
        print("Obteniendo datos recientes...")
        df = obtener_datos_recientes()
        
        if df.empty:
            print("No hay datos suficientes para analizar")
            return
        
        print("Detectando faltas consecutivas recientes...")
        faltas_consecutivas = detectar_faltas_recientes_consecutivas(df)
        
        if not faltas_consecutivas.empty:
            print(f"✔ Se encontraron {len(faltas_consecutivas)} patrones de faltas consecutivas")
            # Guardar predicciones para faltas consecutivas
            for _, row in faltas_consecutivas.iterrows():
                guardar_prediccion(
                    row['id_persona'],
                    row['id_horario'],
                    f"ALERTA: {row['num_faltas']} faltas consecutivas (última: {row['ultima_falta'].date()})"
                )
        else:
            print("✔ No se encontraron patrones de faltas consecutivas recientes")
        
        print("Preparando datos para modelo predictivo...")
        X_train, X_test, y_train, y_test = preparar_datos(df)
        
        print("Entrenando modelo...")
        modelo = RandomForestClassifier(
            n_estimators=100,
            class_weight={0: 2, 1: 1.5, 2: 1},  # Dar más peso a faltas y retardos
            random_state=42
        )
        modelo.fit(X_train, y_train)
        
        print("Generando predicciones para hoy...")
        hoy = datetime.now()
        
        # Obtener combinaciones únicas de profesor/horario con sus estadísticas
        profes_horarios = df[['id_persona', 'id_horario', 'porc_falta', 'porc_retardo']].drop_duplicates()
        
        for _, row in profes_horarios.iterrows():
            # Solo predecir si tiene historial reciente de faltas/retardos
            if row['porc_falta'] > 0.1 or row['porc_retardo'] > 0.1:
                datos_hoy = pd.DataFrame({
                    'dia_semana': [hoy.weekday()],
                    'dia_mes': [hoy.day],
                    'porc_falta': [row['porc_falta']],
                    'porc_retardo': [row['porc_retardo']]
                }, index=[0])
                
                proba = modelo.predict_proba(datos_hoy)[0]
                
                if proba[0] >= UMBRAL_FALTA_ALTA:
                    guardar_prediccion(
                        row['id_persona'],
                        row['id_horario'],
                        "ALTA PROBABILIDAD DE FALTA HOY"
                    )
                elif proba[0] >= UMBRAL_FALTA_MEDIA:
                    guardar_prediccion(
                        row['id_persona'],
                        row['id_horario'],
                        "PROBABLE FALTA HOY"
                    )
                elif proba[1] >= UMBRAL_RETARDO:
                    guardar_prediccion(
                        row['id_persona'],
                        row['id_horario'],
                        "POSIBLE RETARDO HOY"
                    )
        
        print("✅ Proceso de predicción completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error en el proceso de predicción: {str(e)}")
        raise

if __name__ == "__main__":
    ejecutar_predicciones()