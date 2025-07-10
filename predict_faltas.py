import os
import pandas as pd
from urllib.parse import urlparse
from datetime import date, datetime
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Conexión a la base de datos usando MYSQL_URL
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


# Consulta de datos de asistencia
def obtener_datos():
    conexion = conectar_db()
    
    # Obtener el periodo actual (similar a tu lógica en JavaScript)
    fecha_actual = datetime.now()
    anio = fecha_actual.year
    mes = fecha_actual.month
    periodo = 1 if (mes >= 8 or mes <= 1) else 2
    
    # Primero obtener el id_periodo actual
    query_periodo = """
    SELECT id_periodo FROM periodos WHERE anio = %s AND periodo = %s
    """
    cursor = conexion.cursor()
    cursor.execute(query_periodo, (anio, periodo))
    periodo_rows = cursor.fetchall()
    
    if not periodo_rows:
        conexion.close()
        raise Exception('No existe el periodo actual.')
    
    id_periodo = periodo_rows[0][0]
    
    # Obtener el id_contenedor del periodo actual
    query_contenedor = """
    SELECT id_contenedor FROM contenedor WHERE id_periodo = %s
    """
    cursor.execute(query_contenedor, (id_periodo,))
    contenedor_rows = cursor.fetchall()
    
    if not contenedor_rows:
        conexion.close()
        raise Exception('No existe un contenedor para el periodo actual.')
    
    id_contenedor = contenedor_rows[0][0]
    
    # Consulta principal filtrada por el contenedor actual
    query = """
    SELECT
        p.id_persona,
        h.id_horario,
        h.dia_horario,
        h.hora_inicio,
        a.fecha_asistencia,
        CASE
            WHEN a.validacion_asistencia = 1 THEN 2
            WHEN r.validacion_retardo = 1 THEN 1
            WHEN f.validacion_falta = 1 THEN 0
            ELSE NULL
        END AS tipo_asistencia
    FROM horario h
    JOIN persona p ON h.id_persona = p.id_persona
    LEFT JOIN asistencia a ON a.id_horario = h.id_horario
    LEFT JOIN retardo r ON r.id_horario = h.id_horario AND r.fecha_retardo = a.fecha_asistencia
    LEFT JOIN falta f ON f.id_horario = h.id_horario AND f.fecha_falta = a.fecha_asistencia
    WHERE h.id_contenedor = %s
    """
    
    df = pd.read_sql(query, conexion, params=(id_contenedor,))
    cursor.close()
    conexion.close()
    return df
# Obtener id_escuela real a partir del id_horario
def obtener_id_escuela(id_horario):
    conexion = conectar_db()
    cursor = conexion.cursor()
    cursor.execute("SELECT id_escuela FROM horario WHERE id_horario = %s", (id_horario,))
    result = cursor.fetchone()
    cursor.close()
    conexion.close()
    return result[0] if result else None

# Borrar todas las predicciones existentes
def borrar_todas_predicciones():
    conexion = conectar_db()
    cursor = conexion.cursor()
    cursor.execute("DELETE FROM predicciones")
    conexion.commit()
    cursor.close()
    conexion.close()
    print("✔ Todas las predicciones han sido borradas.")

# Guardar una predicción
def guardar_prediccion(id_persona, id_horario, resultado):
    id_persona = int(id_persona)
    id_horario = int(id_horario)
    id_escuela = obtener_id_escuela(id_horario)
    if id_escuela is None:
        print(f" No se encontró el id_escuela para el horario {id_horario}")
        return

    conexion = conectar_db()
    cursor = conexion.cursor()
    cursor.execute("""
        INSERT INTO predicciones (id_persona, id_horario, id_escuela, fecha_prediccion, resultado_prediccion)
        VALUES (%s, %s, %s, %s, %s)
    """, (id_persona, id_horario, int(id_escuela), date.today(), resultado))
    conexion.commit()
    cursor.close()
    conexion.close()
    print(f" Predicción guardada: {resultado} (persona {id_persona}, horario {id_horario}, escuela {id_escuela})")
    
# Preparar datos para el modelo
def preparar_datos(df):
    df = df.dropna(subset=['tipo_asistencia'])
    df['fecha_asistencia'] = pd.to_datetime(df['fecha_asistencia'])
    df['dia_semana'] = df['fecha_asistencia'].dt.dayofweek
    df['mes'] = df['fecha_asistencia'].dt.month

    # Usamos directamente la columna tipo_asistencia (0=falta, 1=retardo, 2=asistencia)
    X = df[['dia_semana', 'mes']]
    y = df['tipo_asistencia']

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Función principal modificada (solo la parte de predicción)
def ejecutar():
    print("Borrando todas las predicciones existentes...")
    borrar_todas_predicciones()

    print("Obteniendo datos de la base de datos...")
    df = obtener_datos()
    if df.empty:
        print("⚠ No hay datos para entrenar el modelo.")
        return

    print("Preparando datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)

    print("🤖 Entrenando modelo multiclase...")
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)

    print("Usando fecha actual para predicción...")
    hoy = datetime.today()
    dia_semana_pred = hoy.weekday()
    mes_pred = hoy.month

    # Obtener todos los horarios únicos
    combinaciones = df[['id_persona', 'id_horario']].drop_duplicates()
    
    # Umbrales para predicciones
    UMBRAL_FALTA = 0.6  # 60% probabilidad de falta
    UMBRAL_RETARDO = 0.4  # 40% probabilidad de retardo

    print("Realizando predicciones para todos los horarios...")
    for _, row in combinaciones.iterrows():
        datos_prediccion = pd.DataFrame([{
            'dia_semana': dia_semana_pred,
            'mes': mes_pred
        }])
        
        # Obtener probabilidades para cada clase [P(falta), P(retardo), P(asistencia)]
        probabilidades = modelo.predict_proba(datos_prediccion)[0]
        
        # Solo guardar faltas y retardos significativos
        if probabilidades[0] >= UMBRAL_FALTA:
            guardar_prediccion(row['id_persona'], row['id_horario'], "FALTARÁ")
        elif probabilidades[1] >= UMBRAL_RETARDO:
            guardar_prediccion(row['id_persona'], row['id_horario'], "POSIBLE RETARDO")
        # No guardamos asistencias previstas

    print("✅ Predicciones guardadas correctamente (solo faltas y retardos significativos)")

if __name__ == "__main__":
    ejecutar()