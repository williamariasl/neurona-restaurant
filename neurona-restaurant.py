import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

# ==========================================
# 1. PREPARACIÓN DE DATOS (Simulación)
# ==========================================
# En la vida real, cargarías un CSV con pandas: 
# df = pd.read_csv('ventas_restaurante.csv')

# Simulamos 1000 ventanas temporales de 30 minutos (aprox. 20 días)
# Variables: [Hora_del_dia, Dia_de_la_semana, Clima(0-1), Ventas_Zona_A, Ventas_Zona_B]
n_registros = 1000
datos_simulados = np.random.rand(n_registros, 5) * 100 

df = pd.DataFrame(datos_simulados, columns=['Hora', 'DiaSemana', 'Clima', 'Demanda_ZonaA', 'Demanda_ZonaB'])

# Escalar los datos entre 0 y 1 (Fundamental para que la red neuronal LSTM aprenda bien)
scaler = MinMaxScaler(feature_range=(0, 1))
datos_escalados = scaler.fit_transform(df)

# ==========================================
# 2. CREACIÓN DE VENTANAS TEMPORALES (Secuencias)
# ==========================================
# Las LSTM necesitan mirar hacia atrás. Le diremos que mire las últimas 4 ventanas 
# (2 horas de datos) para predecir la siguiente ventana de 30 minutos.
def crear_secuencias(datos, pasos_atras):
    X, y = [], []
    for i in range(len(datos) - pasos_atras):
        X.append(datos[i:(i + pasos_atras), :]) # Tomamos todas las variables de las últimas 2 horas
        y.append(datos[i + pasos_atras, 3:])    # Predecimos solo la Demanda (Columnas 3 y 4)
    return np.array(X), np.array(y)

pasos_atras = 4 # 4 ventanas de 30 min = 2 horas de memoria
X, y = crear_secuencias(datos_escalados, pasos_atras)

# Dividir en datos de Entrenamiento (80%) y Prueba (20%)
limite_split = int(len(X) * 0.8)
X_train, X_test = X[:limite_split], X[limite_split:]
y_train, y_test = y[:limite_split], y[limite_split:]

# ==========================================
# 3. DISEÑO DE LA ARQUITECTURA DE LA RED NEURONAL
# ==========================================
model = Sequential()

# Primera capa LSTM (Procesa la secuencia temporal)
# input_shape = (pasos_atras, cantidad_de_variables)
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Previene el sobreajuste (overfitting)

# Segunda capa LSTM (Extrae patrones más profundos)
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Capa de Salida (2 neuronas predictivas: Demanda Zona A y Demanda Zona B)
model.add(Dense(units=2, activation='linear'))

# Compilar el modelo (Usamos MSE como función de pérdida como dicta la teoría)
model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary()) # Muestra la arquitectura en consola

# ==========================================
# 4. ENTRENAMIENTO DEL MODELO (Fase Actuar)
# ==========================================
# Entrenamos la red neuronal con los datos históricos
historial = model.fit(
    X_train, y_train,
    epochs=50,          # Veces que el algoritmo verá todos los datos
    batch_size=16,      # Lotes de datos a procesar por iteración
    validation_split=0.1, 
    verbose=1
)

# ==========================================
# 5. EVALUACIÓN Y REFLEXIÓN (Fase de Comprobación)
# ==========================================
# Hacemos que el modelo intente predecir el futuro con los datos de prueba
predicciones = model.predict(X_test)

# Como los datos estaban escalados de 0 a 1, debemos revertir la escala 
# para leer las métricas en "cantidad de platos reales"
predicciones_reales = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predicciones), 3)), predicciones), axis=1)
)[:, 3:]

y_test_reales = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_test), 3)), y_test), axis=1)
)[:, 3:]

# Calculamos las métricas prometidas en el Entregable 5
mae = mean_absolute_error(y_test_reales, predicciones_reales)
rmse = math.sqrt(mean_squared_error(y_test_reales, predicciones_reales))

print(f"\n--- RESULTADOS DE LA EVALUACIÓN ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} platos de diferencia promedio")
print(f"Error Cuadrático Medio (RMSE): {rmse:.2f} (penaliza errores grandes en horas pico)")