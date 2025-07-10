import os
import pandas as pd
from datetime import date, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Configuración de predicción
UMBRAL_FALTA = 0.65  # Probabilidad mínima para predecir falta
UMBRAL_RETARDO = 0.35  # Probabilidad mínima para predecir retardo

# Conexión a la base de datos usando SQLAlchemy
def conectar_db():
    mysql_url = os.getenv("MYSQL_URL")
    return create_engine(mysql_url)

# Obtener datos históricos de asistencia
def obtener_datos():
    engine = conectar_db()
    
    # Determinar periodo actual
    hoy = datetime.now()
    periodo = 1 if (hoy.month >= 8 or hoy.month <= 1) else 2
    
    query = """
    SELECT
        p.id_persona,
        h.id_horario,
        h.dia_horario,
        h.hora_inicio,
        a.fecha_asistencia,
        CASE
            WHEN a.validacion_asistencia = 1 THEN 2  # Asistencia
            WHEN r.validacion_retardo = 1 THEN 1     # Retardo
            WHEN f.validacion_falta = 1 THEN 0       # Falta
            ELSE NULL
        END AS tipo_asistencia
    FROM horario h
    JOIN persona p ON h.id_persona = p.id_persona
    LEFT JOIN asistencia a ON a.id_horario = h.id_horario
    LEFT JOIN retardo r ON r.id_horario = h.id_horario AND r.fecha_retardo = a.fecha_asistencia
    LEFT JOIN falta f ON f.id_horario = h.id_horario AND f.fecha_falta = a.fecha_asistencia
    WHERE h.id_contenedor = (
        SELECT id_contenedor FROM contenedor WHERE id_periodo = (
            SELECT id_periodo FROM periodos 
            WHERE anio = %s AND periodo = %s
        )
    )
    """
    
    df = pd.read_sql(query, engine, params=(hoy.year, periodo))
    return df

# Obtener id_escuela para un horario
def obtener_id_escuela(id_horario):
    engine = conectar_db()
    result = pd.read_sql(
        "SELECT id_escuela FROM horario WHERE id_horario = %s", 
        engine, params=(id_horario,)
    )
    return result.iloc[0, 0] if not result.empty else None

# Borrar predicciones anteriores
def borrar_predicciones():
    engine = conectar_db()
    with engine.begin() as conn:
        conn.execute("DELETE FROM predicciones")
    print("Predicciones anteriores borradas.")

# Guardar nueva predicción
def guardar_prediccion(id_persona, id_horario, resultado):
    id_escuela = obtener_id_escuela(id_horario)
    if id_escuela is None:
        print(f"⚠ No se encontró escuela para horario {id_horario}")
        return

    engine = conectar_db()
    with engine.begin() as conn:
        conn.execute(
            "INSERT INTO predicciones VALUES (%s, %s, %s, %s, %s)",
            (id_persona, id_horario, id_escuela, date.today(), resultado)
        )
    print(f"✅ Predicción guardada: {resultado} (Persona: {id_persona}, Horario: {id_horario})")

# Preparar datos para el modelo
def preparar_datos(df):
    df = df.dropna(subset=['tipo_asistencia'])
    
    # Convertir fechas y horas
    df['fecha_asistencia'] = pd.to_datetime(df['fecha_asistencia'])
    df['dia_semana'] = df['fecha_asistencia'].dt.dayofweek
    df['mes'] = df['fecha_asistencia'].dt.month
    
    # Convertir hora_inicio a formato numérico (horas)
    if pd.api.types.is_timedelta64_dtype(df['hora_inicio']):
        df['hora_inicio'] = df['hora_inicio'].dt.components['hours']
    else:
        df['hora_inicio'] = pd.to_datetime(df['hora_inicio']).dt.hour

    # Preparar características (X) y variable objetivo (y)
    X = df[['dia_semana', 'mes', 'hora_inicio']]
    y = df['tipo_asistencia']  # 0=Falta, 1=Retardo, 2=Asistencia
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Obtener horarios para el día actual
def obtener_horarios_hoy():
    hoy = datetime.now()
    dia_nombre = hoy.strftime('%A').capitalize()
    periodo = 1 if (hoy.month >= 8 or hoy.month <= 1) else 2
    
    engine = conectar_db()
    query = """
    SELECT h.id_horario, h.id_persona, h.hora_inicio 
    FROM horario h 
    WHERE h.dia_horario = %s AND h.id_contenedor = (
        SELECT id_contenedor FROM contenedor WHERE id_periodo = (
            SELECT id_periodo FROM periodos 
            WHERE anio = %s AND periodo = %s
        )
    )
    """
    return pd.read_sql(query, engine, params=(dia_nombre, hoy.year, periodo))

# Función principal
def ejecutar():
    print("🚀 Iniciando proceso de predicción de asistencia")
    
    # 1. Limpiar predicciones anteriores
    borrar_predicciones()
    
    # 2. Obtener datos históricos
    print("📊 Obteniendo datos históricos...")
    df = obtener_datos()
    if df.empty:
        print("⚠ No hay datos suficientes para entrenar el modelo")
        return
    
    # 3. Preparar datos para el modelo
    print("🧠 Preparando datos para el modelo...")
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    # 4. Entrenar modelo
    print("🤖 Entrenando modelo Random Forest...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    print(f"📈 Precisión del modelo: {modelo.score(X_test, y_test):.2%}")
    
    # 5. Obtener horarios para hoy
    print("🕒 Obteniendo horarios para hoy...")
    horarios_hoy = obtener_horarios_hoy()
    if horarios_hoy.empty:
        print("⚠ No hay clases programadas para hoy")
        return
    
    # 6. Realizar predicciones
    print("🔮 Generando predicciones...")
    hoy = datetime.now()
    total_faltas = 0
    total_retardos = 0
    
    for _, row in horarios_hoy.iterrows():
        # Preparar datos para predicción
        hora_inicio = row['hora_inicio']
        if pd.api.types.is_timedelta64_dtype(horarios_hoy['hora_inicio']):
            hora_num = hora_inicio.components['hours']
        else:
            hora_num = pd.to_datetime(hora_inicio).hour if hora_inicio else 8
        
        datos_pred = pd.DataFrame([{
            'dia_semana': hoy.weekday(),
            'mes': hoy.month,
            'hora_inicio': hora_num
        }])
        
        # Obtener probabilidades
        probas = modelo.predict_proba(datos_pred)[0]
        
        # Tomar decisión basada en umbrales
        if probas[0] >= UMBRAL_FALTA:  # Falta
            guardar_prediccion(row['id_persona'], row['id_horario'], "FALTARÁ")
            total_faltas += 1
        elif probas[1] >= UMBRAL_RETARDO:  # Retardo
            guardar_prediccion(row['id_persona'], row['id_horario'], "POSIBLE RETARDO")
            total_retardos += 1
        # No guardamos asistencias previstas
    
    print(f"\n✅ Proceso completado: {total_faltas} faltas y {total_retardos} retardos predichos")

if __name__ == "__main__":
    ejecutar()