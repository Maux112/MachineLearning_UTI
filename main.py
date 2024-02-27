import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
# Carga los datos desde el archivo CSV
data = pd.read_csv('C:/Users/Mauro/Desktop/TRABAJO/UTI_Statistics/Media/ExcelRespaldoBd/ParIA.csv', encoding='utf-8', sep=',')
# Preprocesamiento de los datos
X = data[['GENERO', 'EDAD', 'ORIGEN DE INGRESO', 'CAUSA DE INGRESO', 'SUBCATEGORIAS DE ORIGEN DE INGRESO']]
y = data['DÍAS DE ESTANCIA']
# Codificación one-hot para las variables categóricas
X_encoded = pd.get_dummies(X)
# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Definir los hiperparámetros a ajustar
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 1000],
    'max_depth': [5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
# Crear un modelo de XGBoost
xgb_model = XGBRegressor(random_state=42)
# Utilizar GridSearchCV para encontrar la mejor combinación de hiperparámetros
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# Obtener el mejor modelo y hacer predicciones
best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)
# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R^2: {r2}')
# Guardar el modelo entrenado
joblib.dump(best_xgb_model, 'modelo_xgboost_entrenado.pkl')
# Graficar las predicciones versus los valores reales
plt.scatter(y_test, y_pred)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores reales de los días de estancia')
plt.show()

print('Fin del Proceso')