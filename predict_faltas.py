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
    result = urlparse(mysql_url)
    return mysql.connector.connect(
        host=result.hostname,
        user=result.username,
        password=result.password,
        database=result.path[1:],
        port=result.port
    )

def obtener_datos_recientes():
    conexion = conectar_db()
    
    # Calcular fecha límite (hoy - DIAS_RECIENTES)
    fecha_limite = (datetime.now() - timedelta(days=DIAS_RECIENTES)).strftime('%Y-%m-%d')
    
    # Obtener periodo actual
    fecha_actual = datetime.now()
    anio = fecha_actual.year
    mes = fecha_actual.month
    periodo = 1 if (mes >= 8 or mes <= 1) else 2
    
    query_periodo = "SELECT id_periodo FROM periodos WHERE anio = %s AND periodo = %s"
    cursor = conexion.cursor()
    cursor.execute(query_periodo, (anio, periodo))
    id_periodo = cursor.fetchone()[0] if cursor.fetchone() else None
    
    if not id_periodo:
        conexion.close()
        raise Exception('No existe el periodo actual.')
    
    # Obtener contenedor actual
    cursor.execute("SELECT id_contenedor FROM contenedor WHERE id_periodo = %s", (id_periodo,))
    id_contenedor = cursor.fetchone()[0] if cursor.fetchone() else None
    
    if not id_contenedor:
        conexion.close()
        raise Exception('No existe contenedor para el periodo actual.')
    
    # Consulta optimizada para datos recientes
    query = f"""
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
    LEFT JOIN asistencia a ON a.id_horario = h.id_horario AND a.fecha_asistencia >= '{fecha_limite}'
    LEFT JOIN retardo r ON r.id_horario = h.id_horario AND r.fecha_retardo = a.fecha_asistencia
    LEFT JOIN falta f ON f.id_horario = h.id_horario AND f.fecha_falta = a.fecha_asistencia
    WHERE h.id_contenedor = %s
    ORDER BY p.id_persona, h.id_horario, a.fecha_asistencia DESC
    """
    
    df = pd.read_sql(query, conexion, params=(id_contenedor,))
    conexion.close()
    return df

def detectar_faltas_recientes_consecutivas(df):
    # Filtrar solo faltas (tipo_asistencia = 0)
    df_faltas = df[df['tipo_asistencia'] == 0].copy()
    
    # Convertir a datetime y ordenar
    df_faltas['fecha_asistencia'] = pd.to_datetime(df_faltas['fecha_asistencia'])
    df_faltas = df_faltas.sort_values(['id_persona', 'id_horario', 'fecha_asistencia'])
    
    # Calcular diferencia entre fechas consecutivas
    df_faltas['diff_dias'] = df_faltas.groupby(['id_persona', 'id_horario'])['fecha_asistencia'].diff().dt.days
    
    # Identificar secuencias de faltas consecutivas (máximo 7 días entre faltas)
    df_faltas['consecutivo'] = (df_faltas['diff_dias'] != 1).cumsum()
    
    # Contar faltas consecutivas por grupo
    conteo_faltas = df_faltas.groupby(['id_persona', 'id_horario', 'consecutivo']).size().reset_index(name='num_faltas')
    
    # Filtrar solo los que tienen MIN_FALTAS_CONSECUTIVAS o más
    faltas_significativas = conteo_faltas[conteo_faltas['num_faltas'] >= MIN_FALTAS_CONSECUTIVAS]
    
    # Obtener las fechas más recientes de cada grupo
    faltas_recientes = df_faltas.groupby(['id_persona', 'id_horario']).first().reset_index()
    
    # Combinar con faltas significativas
    resultado = pd.merge(
        faltas_significativas, 
        faltas_recientes, 
        on=['id_persona', 'id_horario', 'consecutivo']
    )
    
    return resultado[['id_persona', 'id_horario', 'num_faltas', 'fecha_asistencia']]

def preparar_datos(df):
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

def guardar_prediccion(id_persona, id_horario, resultado):
    conexion = conectar_db()
    cursor = conexion.cursor()
    
    # Obtener id_escuela
    cursor.execute("SELECT id_escuela FROM horario WHERE id_horario = %s", (id_horario,))
    id_escuela = cursor.fetchone()[0] if cursor.fetchone() else None
    
    if id_escuela:
        cursor.execute("""
            INSERT INTO predicciones 
            (id_persona, id_horario, id_escuela, fecha_prediccion, resultado_prediccion)
            VALUES (%s, %s, %s, %s, %s)
        """, (int(id_persona), int(id_horario), int(id_escuela), date.today(), resultado))
        conexion.commit()
        print(f"✔ Predicción guardada: {id_persona} - {resultado}")
    else:
        print(f"✖ No se encontró escuela para horario {id_horario}")
    
    cursor.close()
    conexion.close()

def ejecutar_predicciones():
    print("Obteniendo datos recientes...")
    df = obtener_datos_recientes()
    
    if df.empty:
        print("No hay datos suficientes para analizar")
        return
    
    print("Detectando faltas consecutivas recientes...")
    faltas_consecutivas = detectar_faltas_recientes_consecutivas(df)
    
    # Guardar predicciones para faltas consecutivas
    for _, row in faltas_consecutivas.iterrows():
        guardar_prediccion(
            row['id_persona'],
            row['id_horario'],
            f"ALERTA: {row['num_faltas']} faltas consecutivas recientes (última: {row['fecha_asistencia'].date()})"
        )
    
    print("Preparando datos para modelo predictivo...")
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    print("Entrenando modelo...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        class_weight={0: 2, 1: 1.5, 2: 1}  # Dar más peso a faltas y retardos
    )
    modelo.fit(X_train, y_train)
    
    print("Generando predicciones para hoy...")
    hoy = datetime.now()
    datos_hoy = pd.DataFrame({
        'dia_semana': [hoy.weekday()],
        'dia_mes': [hoy.day],
        'porc_falta': [0],  # Se actualizará por profesor
        'porc_retardo': [0]  # Se actualizará por profesor
    })
    
    # Obtener combinaciones únicas de profesor/horario
    profes_horarios = df[['id_persona', 'id_horario', 'porc_falta', 'porc_retardo']].drop_duplicates()
    
    for _, row in profes_horarios.iterrows():
        # Solo predecir si tiene historial reciente de faltas/retardos
        if row['porc_falta'] > 0.1 or row['porc_retardo'] > 0.1:
            datos_hoy['porc_falta'] = row['porc_falta']
            datos_hoy['porc_retardo'] = row['porc_retardo']
            
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

if __name__ == "__main__":
    # Limpiar predicciones anteriores
    conexion = conectar_db()
    cursor = conexion.cursor()
    cursor.execute("DELETE FROM predicciones")
    conexion.commit()
    cursor.close()
    conexion.close()
    
    ejecutar_predicciones()