# Abner Maximiliano Lecona Nieves
# A01753179

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Lectura del archivo
data = pd.read_csv("500hits.csv", encoding='latin-1')

# Gráfico de barras para la variable objetivo
data['HOF'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de la Variable Objetivo (HOF)')
plt.xlabel('HOF (1 = Entra, 0 = No entra)')
plt.ylabel('Número de Jugadores')
plt.show()

# quitar el atributo PLAYER y CS, no serviran
data = data.drop(columns= ['PLAYER', 'CS'] )

# Separar las características (X) y la variable objetivo (y)
X = data.iloc[:, :-1]
y = data.iloc[:,-1] 

# Separar los datos en train y test
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42)

# Visualizacion de training y test
# Mostrar la forma de los conjuntos
print(f'Tamaño del conjunto de entrenamiento: {x_train.shape}')
print(f'Tamaño del conjunto de prueba: {x_test.shape}')

# Gráficos de distribución 
plt.figure(figsize=(10, 6))
sns.histplot(x_train['H'], color='blue', label='Entrenamiento', kde=True)
sns.histplot(x_test['H'], color='red', label='Prueba', kde=True)
plt.title('Distribución de Hits (H) en Train/Test')
plt.xlabel('Número de Hits (H)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

#RandomForest
forest_class = RandomForestClassifier()

# Función para graficar curvas de aprendizaje
def plot_learning_curve(estimator, X, y, title="Curva de Aprendizaje"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Error en Entrenamiento")
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label="Error en Validación")
    plt.title(title)
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Graficar la curva de aprendizaje
plot_learning_curve(forest_class, x_train, y_train, title="Curva de Aprendizaje para Random Forest")

# Entrenar y prueba del modelo
forest_class.fit(x_train, y_train)
y_pred =forest_class.predict(x_test)

# Evaluar el rendimiento del modelo
train_score = forest_class.score(x_train, y_train)
test_score = forest_class.score(x_test, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f"\nDatos del modelo con datos")
print(f"Training Score: {train_score:.2f}")
print(f"Test Score: {test_score:.2f}")

print(f"\n{classification_report(y_test, y_pred)}\n")

print("Matriz de Confusión:")
print(conf_matrix)

# Diagnóstico basado en las curvas de aprendizaje
if abs(train_score - test_score) < 0.05:
    print("El modelo está bien ajustado (fit).")
elif train_score > test_score:
    print("El modelo muestra un posible sobreajuste (overfitting).")
else:
    print("El modelo muestra un posible subajuste (underfitting).")

'''
---------------------------------------------------------------------------------
'''
# Model con los parametros encontrados del grid #
print("\nResultados del Gird Search\n")
print("randomizing...")

# Definir el modelo
model_random = RandomForestClassifier()

# Definir el grid de hiperparámetros
param_dist = {
    'n_estimators': [25, 50, 75],  # Número de árboles en el bosque
    'max_depth': [3, 6, 9],  # Máxima profundidad del árbol
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras para dividir un nodo
}

# Crear el objeto RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model_random,
    param_distributions=param_dist,
    n_iter=50,  # Número de combinaciones a probar
    cv=5,  # Número de pliegues de validación cruzada
    scoring='accuracy',  # Métrica de evaluación
    verbose=1,
    random_state=42,  # Fijar la semilla para la reproducibilidad
    n_jobs=-1  # Usar todos los núcleos disponibles
)

# Ajustar RandomizedSearchCV
random_search.fit(x_train, y_train)

# Imprimir los mejores parámetros y el mejor score
print("Mejores parámetros randomizados:", random_search.best_params_)
print("Mejor score:", random_search.best_score_)

#Mejores parámetros randomizados: {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 9}
'''
--------------------------------------------------------------------------------------------
'''
print("griding...")
model_grid = RandomForestClassifier()
# Definir el grid de hiperparámetros
param_grid = {
    'n_estimators': [45, 50, 55],  # Número de árboles en el bosque
    'max_depth': [5, 6, 7],  # Máxima profundidad del árbol
    'min_samples_split': [8, 10, 12], 
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(estimator=model_grid, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Ajustar GridSearchCV
grid_search.fit(x_train, y_train)

# Imprimir los mejores parámetros y el mejor score
print("Mejores parámetros despues de gridear:", grid_search.best_params_)
# Mejores parámetros despues del random y grid: {'max_depth': 7, 'n_estimators': 50}
'''
------------------------------------------------------------------------------------------------
'''
# Obtener los mejores parámetros encontrados por RandomizedSearchCV
best_params = grid_search.best_params_

# Imprimir los mejores parámetros
print("Mejores parámetros despues del random y grid:", best_params)
# Mejores parámetros despues del random y grid: {'max_depth': 7, 'n_estimators': 50}

# Crear un nuevo modelo de RandomForestClassifier con los mejores parámetros
best_model = RandomForestClassifier(**best_params)
'''
------------------------------------------------------------------------------------------------
'''
# Función para graficar curvas de aprendizaje
def plot_learning_curve(estimator, X, y, title="Curva de Aprendizaje"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Error en Entrenamiento")
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label="Error en Validación")
    plt.title(title)
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Graficar la curva de aprendizaje
plot_learning_curve(best_model, x_train, y_train, title="Curva de Aprendizaje para Random Forest")

# Entrenar y prueba del modelo
best_model.fit(x_train, y_train)
y_pred =best_model.predict(x_test)

# Evaluar el rendimiento del modelo
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f"\nDatos del modelo con datos")
print(f"Training Score: {train_score:.2f}")
print(f"Test Score: {test_score:.2f}")

print(f"\n{classification_report(y_test, y_pred)}\n")

print("Matriz de Confusión:")
print(conf_matrix)

# Diagnóstico basado en las curvas de aprendizaje
if abs(train_score - test_score) < 0.05:
    print("El modelo está bien ajustado (fit).")
elif train_score > test_score:
    print("El modelo muestra un posible sobreajuste (overfitting).")
else:
    print("El modelo muestra un posible subajuste (underfitting).")