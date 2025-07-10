import os
import pandas as pd
from urllib.parse import urlparse
from datetime import date, datetime, time
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def conectar_db():
    mysql_url = os.getenv("MYSQL_URL")
    result = urlparse(mysql_url)
    
    # Usamos SQLAlchemy para mayor compatibilidad
    from sqlalchemy import create_engine
    db_url = f"mysql+mysqlconnector://{result.username}:{result.password}@{result.hostname}:{result.port}{result.path}"
    return create_engine(db_url)

# Consulta de datos de asistencia histórica
def obtener_datos():
    engine = conectar_db()
    
    # Obtener el periodo actual
    fecha_actual = datetime.now()
    anio = fecha_actual.year
    mes = fecha_actual.month
    periodo = 1 if (mes >= 8 or mes <= 1) else 2
    
    # Consulta para obtener el contenedor actual
    query = """
    SELECT h.id_persona, h.id_horario, h.dia_horario, h.hora_inicio, 
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
    WHERE h.id_contenedor = (
        SELECT id_contenedor FROM contenedor WHERE id_periodo = (
            SELECT id_periodo FROM periodos WHERE anio = %s AND periodo = %s
        )
    )
    """
    
    df = pd.read_sql(query, engine, params=(anio, periodo))
    return df

# Obtener id_escuela
def obtener_id_escuela(id_horario):
    engine = conectar_db()
    result = pd.read_sql("SELECT id_escuela FROM horario WHERE id_horario = %s", 
                        engine, params=(id_horario,))
    return result.iloc[0, 0] if not result.empty else None

# Borrar predicciones
def borrar_todas_predicciones():
    engine = conectar_db()
    with engine.begin() as conn:
        conn.execute("DELETE FROM predicciones")
    print("✔ Todas las predicciones han sido borradas.")

# Guardar predicción
def guardar_prediccion(id_persona, id_horario, resultado):
    id_escuela = obtener_id_escuela(id_horario)
    if id_escuela is None:
        print(f"No se encontró escuela para horario {id_horario}")
        return

    engine = conectar_db()
    with engine.begin() as conn:
        conn.execute(
            "INSERT INTO predicciones VALUES (%s, %s, %s, %s, %s)",
            (id_persona, id_horario, id_escuela, date.today(), resultado)
        )
    print(f"Predicción guardada: {resultado} (persona {id_persona}, horario {id_horario})")

# Preparar datos
def preparar_datos(df):
    df = df.dropna(subset=['tipo_asistencia'])
    df['fecha_asistencia'] = pd.to_datetime(df['fecha_asistencia'])
    df['dia_semana'] = df['fecha_asistencia'].dt.dayofweek
    df['mes'] = df['fecha_asistencia'].dt.month
    
    # Convertir hora_inicio (timedelta) a horas
    if pd.api.types.is_timedelta64_dtype(df['hora_inicio']):
        df['hora_inicio'] = df['hora_inicio'].dt.components['hours']
    else:
        # Si es string, extraer la hora
        df['hora_inicio'] = pd.to_datetime(df['hora_inicio']).dt.hour

    X = df[['dia_semana', 'mes', 'hora_inicio']]
    y = df['tipo_asistencia']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Función principal
def ejecutar():
    print("Borrando predicciones existentes...")
    borrar_todas_predicciones()

    print("Obteniendo datos históricos...")
    df = obtener_datos()
    if df.empty:
        print("⚠ No hay datos para entrenar")
        return

    print("Preparando datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)

    print("Entrenando modelo...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    print(f"Precisión: {modelo.score(X_test, y_test):.2%}")

    print("Generando predicciones para hoy...")
    hoy = datetime.now()
    dia_nombre = hoy.strftime('%A').capitalize()
    
    # Obtener horarios de hoy
    engine = conectar_db()
    query_horarios = """
    SELECT h.id_horario, h.id_persona, h.hora_inicio 
    FROM horario h 
    WHERE h.dia_horario = %s AND h.id_contenedor = (
        SELECT id_contenedor FROM contenedor WHERE id_periodo = (
            SELECT id_periodo FROM periodos 
            WHERE anio = %s AND periodo = %s
        )
    )
    """
    periodo = 1 if (hoy.month >= 8 or hoy.month <= 1) else 2
    horarios_hoy = pd.read_sql(query_horarios, engine, 
                             params=(dia_nombre, hoy.year, periodo))

    # Procesar cada horario
    for _, row in horarios_hoy.iterrows():
        hora_inicio = row['hora_inicio']
        
        # Convertir hora_inicio a número de horas
        if pd.api.types.is_timedelta64_dtype(horarios_hoy['hora_inicio']):
            hora_num = hora_inicio.components['hours']
        else:
            hora_num = pd.to_datetime(hora_inicio).hour if hora_inicio else 8
        
        datos_pred = pd.DataFrame([{
            'dia_semana': hoy.weekday(),
            'mes': hoy.month,
            'hora_inicio': hora_num
        }])
        
        probas = modelo.predict_proba(datos_pred)[0]
        
        # Umbrales
        if probas[0] >= 0.6:  # Falta
            guardar_prediccion(row['id_persona'], row['id_horario'], "FALTARÁ")
        elif probas[1] >= 0.4:  # Retardo
            guardar_prediccion(row['id_persona'], row['id_horario'], "POSIBLE RETARDO")

    print("Proceso completado")

if __name__ == "__main__":
    ejecutar()