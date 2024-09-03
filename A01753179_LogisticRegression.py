# Abner Maximiliano Lecona Nieves
# A01753179
# Regresion Logistica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de hipótesis para regresión logística
def hipotesis(params, sample):
    z = np.dot(params, sample)
    return sigmoid(z)

# Normalización de datos (min max)
def normalizacion(samples):
    samples = np.array(samples, dtype=float)
    min_vals = np.min(samples[:, 1:], axis=0)  # Excluir la primera columna (término de sesgo)
    max_vals = np.max(samples[:, 1:], axis=0)
    range_vals = max_vals - min_vals
    range_vals = np.clip(range_vals, a_min=1e-8, a_max=None)  # Evitar división por cero
    normalized_samples = samples.copy()
    normalized_samples[:, 1:] = (samples[:, 1:] - min_vals) / range_vals
    return normalized_samples.tolist(), min_vals, range_vals

def normalizar_nuevos_datos(nuevos_samples, min_vals, range_vals):
    nuevos_samples = np.array(nuevos_samples, dtype=float)
    normalizados = (nuevos_samples - min_vals) / range_vals
    return normalizados.tolist()

# Función para predecir
def predict(features, weights, min_vals, range_vals):
    # Normalizar características nuevas (excluyendo el término de sesgo)
    features = normalizar_nuevos_datos([features], min_vals, range_vals)[0]
    features = [1] + features  # Añadir el término de sesgo (bias)
    prob = hipotesis(weights, features)
    return 1 if prob >= 0.5 else 0

# Función de descenso de gradiente
def gradient_descent(params, samples, learn_rate, valor_y):
    m = len(samples)
    avance = np.array(params, dtype=float)
    for j in range(len(params)):
        sum_error = 0
        for i in range(m):
            error = hipotesis(params, samples[i]) - valor_y[i]
            sum_error += error * samples[i][j]
        avance[j] = params[j] - (learn_rate / m) * sum_error
    return avance

# Función de costo 
def logistic_cost(params, samples, valor_y):
    m = len(samples)
    total_cost = 0
    for i in range(m):
        prediction = hipotesis(params, samples[i])
        # Clip para asegurar que prediction no sea 0 ni 1
        prediction = np.clip(prediction, 1e-10, 1 - 1e-10)
        total_cost += -valor_y[i] * np.log(prediction) - (1 - valor_y[i]) * np.log(1 - prediction)
    return total_cost / m

# Función de entrenamiento de regresión logística
def logistic_regression(params, samples, valor_y, num_epochs, learning_rate):
    # Añadir el término de sesgo (bias) como una característica adicional con valor 1
    for i in range(len(samples)):
        samples[i] = [1] + samples[i]

    samples, min_vals, range_vals = normalizacion(samples)

    errores = []
    epochs = 0
    while True:
        oldparams = np.array(params, dtype=float)
        params = gradient_descent(params, samples, learning_rate, valor_y)
        costo = logistic_cost(params, samples, valor_y)
        errores.append(costo)

        epochs += 1
        if np.allclose(oldparams, params, atol=1e-6) or epochs >= num_epochs:
            break

    plt.plot(errores)
    plt.xlabel('Épocas')
    plt.ylabel('Costo')
    plt.title('Costo vs Épocas')
    plt.show()
    
    return params, min_vals, range_vals

# Función para calcular métricas manualmente
def calcular_métricas(y_real, y_pred):
    TP = sum((y_real[i] == 1) and (y_pred[i] == 1) for i in range(len(y_real)))  # Verdaderos positivos
    TN = sum((y_real[i] == 0) and (y_pred[i] == 0) for i in range(len(y_real)))  # Verdaderos negativos
    FP = sum((y_real[i] == 0) and (y_pred[i] == 1) for i in range(len(y_real)))  # Falsos positivos
    FN = sum((y_real[i] == 1) and (y_pred[i] == 0) for i in range(len(y_real)))  # Falsos negativos

    # Cálculo de precisión, recall y F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Matriz de confusión
    conf_matrix = np.array([[TN, FP], [FN, TP]])

    return precision, recall, f1, conf_matrix

# Evaluación del modelo
def evaluate_model(params, samples, valor_y, min_vals, range_vals):
    predictions = []
    for i, sample in enumerate(samples):
        print(f"Muestra {i+1} antes de la predicción: {sample}")
        prediction = predict(sample, params, min_vals, range_vals)
        predictions.append(prediction)

    # Calcular métricas manualmente
    precision, recall, f1, conf_matrix = calcular_métricas(valor_y, predictions)

    print(f"Precisión: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Matriz de Confusión:")
    print(conf_matrix)

if __name__ == "__main__":
    # Leer los archivos CSV
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    # Extraer características para entrenamiento y validación
    X_train = train_data[['Horas_estudio', 'Asistencia_clases', 'Calificaciones_anteriores']].values.tolist()
    y_train = train_data['Pasar_examen'].values

    X_val = val_data[['Horas_estudio', 'Asistencia_clases', 'Calificaciones_anteriores']].values.tolist()
    y_val = val_data['Pasar_examen'].values

    # Extraer solo características del conjunto de prueba
    X_test = test_data[['Horas_estudio', 'Asistencia_clases', 'Calificaciones_anteriores']].values.tolist()

    # Parámetros de entrenamiento
    learning_rate = 0.01
    params = [0.0] * (len(X_train[0]) + 1)  # Inicializar parámetros con ceros, incluyendo el bias

    # Entrenamiento del modelo
    params_finales, min_vals, range_vals = logistic_regression(params, X_train, y_train, 50000, learning_rate)
    print("se realizo el entrenamiento")

    # Evaluación del modelo en el conjunto de validación
    print("Evaluación en el conjunto de validación:")
    evaluate_model(params_finales, X_val, y_val, min_vals, range_vals)
    print("Se realizo la evaluacion")


    # Realizar predicciones en el conjunto de prueba y mostrar los resultados
    print("\nPredicciones para el conjunto de prueba:")
    for i, sample in enumerate(X_test):
        prediccion = predict(sample, params_finales, min_vals, range_vals)
        estado = "Aprobado" if prediccion == 1 else "Reprobado"
        print(f"Registro {i + 1}: Predicción = {estado}")
    
    print("Se realizaron todas las operaciones")

