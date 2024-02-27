import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Carga los datos de entrenamiento originales
data = pd.read_csv('C:/Users/Mauro/Desktop/TRABAJO/UTI_Statistics/Media/ExcelRespaldoBd/ParIA.csv', encoding='utf-8', sep=',')

# Definir X como los datos originales
X = data[['GENERO', 'EDAD', 'ORIGEN DE INGRESO', 'CAUSA DE INGRESO', 'SUBCATEGORIAS DE ORIGEN DE INGRESO']]

# Nuevos datos para la predicción
nuevos_datos = pd.DataFrame([
    ['FEMENINO',80, 'EMERGENCIAS', 'NEUROLÓGICO', 'TEC']
], columns=['GENERO', 'EDAD', 'ORIGEN DE INGRESO', 'CAUSA DE INGRESO', 'SUBCATEGORIAS DE ORIGEN DE INGRESO'])

# Carga el modelo entrenado
loaded_model = joblib.load('modelo_xgboost_entrenado.pkl')

# Codificación one-hot y escalado de los nuevos datos
X_combined = pd.concat([X, nuevos_datos])
X_combined_encoded = pd.get_dummies(X_combined)
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined_encoded)
nuevos_datos_scaled = X_combined_scaled[-len(nuevos_datos):]

# Realizar la predicción
prediccion = loaded_model.predict(nuevos_datos_scaled)

print(f'La predicción de días de estancia es: {prediccion}')

