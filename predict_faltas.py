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
    """
    df = pd.read_sql(query, conexion)
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

    X = df[['dia_semana', 'mes']]
    y = (df['tipo_asistencia'] == 0).astype(int)  # 1 = falta, 0 = no falta

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Función principal
# Función principal
def ejecutar():
    print("🧹 Borrando todas las predicciones existentes...")
    borrar_todas_predicciones()

    print("📥 Obteniendo datos de la base de datos...")
    df = obtener_datos()
    if df.empty:
        print("⚠ No hay datos para entrenar el modelo.")
        return

    print("📊 Preparando datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)

    print("🤖 Entrenando modelo...")
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)

    print("📅 Usando fecha actual para predicción...")
    hoy = datetime.today()
    dia_semana_pred = hoy.weekday()  # lunes = 0
    mes_pred = hoy.month

    combinaciones = df[['id_persona', 'id_horario']].drop_duplicates()
    
    predicciones_guardadas = 0

    for _, row in combinaciones.iterrows():
        id_persona = row['id_persona']
        id_horario = row['id_horario']

        nueva_clase = pd.DataFrame([{
            'dia_semana': dia_semana_pred,
            'mes': mes_pred
        }])

        proba_falta = modelo.predict_proba(nueva_clase)[0][1]  # probabilidad de falta

        # Solo guardar predicciones confiables
        if proba_falta >= 0.9:
            guardar_prediccion(id_persona, id_horario, "FALTARÁ")
            predicciones_guardadas += 1
        elif proba_falta <= 0.1:
            guardar_prediccion(id_persona, id_horario, "NO FALTARÁ")
            predicciones_guardadas += 1
        # Si no es confiable, no guardamos nada

    print(f"✅ Total de predicciones confiables guardadas: {predicciones_guardadas}")

    print("Predicciones guardadas correctamente.")

# Ejecutar si se llama directamente
if __name__ == "__main__":
    ejecutar()
