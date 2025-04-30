import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

"""## K-Nearest Neighbors (knn)"""

data = pd.read_csv('weather_forecast_data.csv') # leemos el csv

#print(data)

# Convertir 'rain'/'no rain' a 1/0
data['Rain'] = data['Rain'].apply(lambda x: 1 if x == 'rain' else 0)

# Separar características (X) y etiquetas (y)
X = data.drop('Rain', axis=1)
y = data['Rain']

#Test va a ser el 20% y el train va a ser el 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Modelo de Clasificación: K-Nearest Neighbors ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

#print(X_train)

accuracy = accuracy_score(y_test, y_pred_knn) # Calculamos el acuracy
print(f"Accuracy del modelo KNN: {accuracy:.2f}")

# Predicciones
y_pred = knn.predict(X_test)

# Evaluación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Lluvia', 'Lluvia']))

# Visualización
plt.figure(figsize=(12, 8))

# Convertimos a array numpy para mlxtend porque no vienen como arrays de np por defecto en el dataset
X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test

# Seleccionamos solo dos características para hacer una visualización reducida como ejemplo
feature_idx = [0, 1]  # Primeras dos características
X_vis = X_test_array[:, feature_idx]

# Entrenamos modelo reducido solo con estas features para visualización
knn_vis = KNeighborsClassifier(n_neighbors=3)
knn_vis.fit(X_train.iloc[:, feature_idx], y_train)

plot_decision_regions(X_vis, y_test_array, clf=knn_vis, legend=2)
plt.xlabel(X_train.columns[feature_idx[0]])
plt.ylabel(X_train.columns[feature_idx[1]])
plt.title("Regiones de Decisión KNN (Temperatura vs Humedad)")
plt.show()

"""## Arbol de decisión (decision Trees)"""

#Ejemplo de arbol de decision

dt= DecisionTreeClassifier(
    max_depth=3,
    random_state=42

)

dt.fit(X_train, y_train)

print("Reporte de clasificacion")
print(classification_report(y_test, y_pred))

y_pred_tree= dt.predict(X_test)

print("Reporte de clasificacion")
print(classification_report(y_test, y_pred_tree))

plt.figure(figsize=(15,10))
plot_tree(dt,
          filled=True,
          feature_names=X_train.columns.tolist(),  # nombres reales
          class_names=['No Lluvia', 'Lluvia'])    # etiquetas  para 0 y 1 (no lluvia, lluvia)
plt.show()

"""## Gráfico de Feature Importance"""

importances = dt.feature_importances_
features = X_train.columns
sorted_indices = np.argsort(importances)[::-1]  # Orden descendente

plt.figure(figsize=(12, 6))
plt.bar(
    range(len(features)),
    importances[sorted_indices],
    align='center',
    color='blue',
    edgecolor='black'
)
plt.title('Feature Importance', pad=20, fontsize=14)
plt.xticks(
    range(len(features)),
    features[sorted_indices],
    rotation=45,
    ha='right'
)
plt.ylabel('Importancia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import seaborn as sns

# Como era raro que no aparecieran en el grafico, usamos  para revisar los valores.

# Correlación con la variable objetivo
correlaciones = data.corr()["Rain"].abs().sort_values(ascending=False)
print("Correlación con 'Rain':\n", correlaciones)

# Y vemos que PRessure y Wind_speed tienen muy baja importancia, (de hecho en el grafico, si hacemos zoom se alcanzan a ver dos pequeñas lineas azules)

"""## Modelo de Regresión (Regresión Logística)"""

## Modelo de Regresión (Regresión Logística)
print("\n--- Modelo de Regresión: Regresión Logística ---")
# Nota: Para predecir probabilidades (0-1) usamos regresión logística
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print("\nReporte de clasificación para Regresión Logística:")
print(classification_report(y_test, y_pred_lr))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_lr))
print(f"Precisión: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"MSE (para probabilidades): {mean_squared_error(y_test, y_prob_lr):.4f}")

"""## Comparación de modelos"""

# Comparación de modelos
print("\n--- Comparación de Modelos ---")
print(f"KNN Precisión: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Árbol Decisión Precisión: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"Regresión Logística Precisión: {accuracy_score(y_test, y_pred_lr):.2f}")