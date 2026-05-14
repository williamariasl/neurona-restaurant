import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt # NUEVO: Para graficar los resultados
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D # NUEVO: Capas CNN

# ==========================================
# 1. PREPARACIÓN DE DATOS 
# ==========================================
# Simulamos 1000 ventanas temporales de 30 minutos
n_registros = 1000
datos_simulados = np.random.rand(n_registros, 5) * 100 

df = pd.DataFrame(datos_simulados, columns=['Hora', 'DiaSemana', 'Clima', 'Demanda_ZonaA', 'Demanda_ZonaB'])

# Escalar los datos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
datos_escalados = scaler.fit_transform(df)

# ==========================================
# 2. CREACIÓN DE VENTANAS TEMPORALES
# ==========================================
def crear_secuencias(datos, pasos_atras):
    X, y = [], []
    for i in range(len(datos) - pasos_atras):
        X.append(datos[i:(i + pasos_atras), :]) 
        y.append(datos[i + pasos_atras, 3:])    
    return np.array(X), np.array(y)

pasos_atras = 4 # 4 ventanas de 30 min = 2 horas
X, y = crear_secuencias(datos_escalados, pasos_atras)

limite_split = int(len(X) * 0.8)
X_train, X_test = X[:limite_split], X[limite_split:]
y_train, y_test = y[:limite_split], y[limite_split:]

# ==========================================
# 3. DISEÑO DE LA ARQUITECTURA CNN-LSTM
# ==========================================
model = Sequential()

# Bloque CNN (Extrae características locales y espaciales)
model.add(Conv1D(
    filters=64, 
    kernel_size=2, 
    padding='same', # Mantiene la longitud de la secuencia
    activation='relu', 
    input_shape=(X_train.shape[1], X_train.shape[2])
))
# Reduce la dimensionalidad temporal a la mitad (de 4 pasos a 2 pasos) para condensar la información
model.add(MaxPooling1D(pool_size=2))

# Bloque LSTM (Analiza la secuencia de las características que extrajo la CNN)
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Capa de Salida (Regresión)
model.add(Dense(units=2, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())

# ==========================================
# 4. ENTRENAMIENTO DEL MODELO 
# ==========================================
historial = model.fit(
    X_train, y_train,
    epochs=50,          
    batch_size=16,      
    validation_split=0.1, 
    verbose=1
)

# ==========================================
# 5. EVALUACIÓN Y GRÁFICOS
# ==========================================
predicciones = model.predict(X_test)

predicciones_reales = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predicciones), 3)), predicciones), axis=1)
)[:, 3:]

y_test_reales = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_test), 3)), y_test), axis=1)
)[:, 3:]

mae = mean_absolute_error(y_test_reales, predicciones_reales)
rmse = math.sqrt(mean_squared_error(y_test_reales, predicciones_reales))

print(f"\n--- RESULTADOS DE LA EVALUACIÓN ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} platos")
print(f"Error Cuadrático Medio (RMSE): {rmse:.2f}")

#Genera gráfica de la función de pérdida (Loss)
plt.figure(figsize=(8, 5))
plt.plot(historial.history['loss'], label='Pérdida Entrenamiento (Train)')
plt.plot(historial.history['val_loss'], label='Pérdida Validación (Validation)')
plt.title('Curva de Aprendizaje del Modelo CNN-LSTM')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.xlabel('Épocas')
plt.legend()
plt.grid(True)
plt.show()