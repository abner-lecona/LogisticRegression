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
    min_vals = np.array(min_vals, dtype=float)
    range_vals = np.array(range_vals, dtype=float)
    normalizados = (nuevos_samples - min_vals) / range_vals
    return normalizados.tolist()

# Función para predecir
def predict(features, weights, min_vals, range_vals):
    features = normalizar_nuevos_datos([features], min_vals, range_vals)[0]
    if len(features) != len(weights):
        features = [1] + features  # Agregar el Bias si es necesario
    prob = hipotesis(weights, features)
    return 1 if prob >= 0.5 else 0

# Función de descenso de gradiente
def gradient_descent(params, samples, learning_rate, valor_y):
    m = len(samples)
    for i in range(m):
        error = hipotesis(params, samples[i]) - valor_y[i]
        for j in range(len(params)):
            params[j] -= learning_rate * error * samples[i][j]
    return params

# Función de costo logístico
def logistic_cost(params, samples, valor_y):
    m = len(samples)
    total_cost = 0
    for i in range(m):
        h = hipotesis(params, samples[i])
        total_cost += -valor_y[i] * np.log(h) - (1 - valor_y[i]) * np.log(1 - h)
    return total_cost / m

# Función de entrenamiento de regresión logística
def logistic_regression(params, samples, valor_y, num_epochs, learning_rate):
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

    # Extraer características y etiquetas
    X_train = train_data[['Horas_estudio', 'Asistencia_clases', 'Calificaciones_anteriores']].values.tolist()
    y_train = train_data['Pasar_examen'].values.tolist()

    X_val = val_data[['Horas_estudio', 'Asistencia_clases', 'Calificaciones_anteriores']].values.tolist()
    y_val = val_data['Pasar_examen'].values.tolist()

    X_test = test_data[['Horas_estudio', 'Asistencia_clases', 'Calificaciones_anteriores']].values.tolist()
    y_test = test_data['Pasar_examen'].values.tolist()

    learning_rate = 0.01
    params = [0.0] * (len(X_train[0]) + 1)  # Inicializar parámetros con ceros, incluyendo el bias

    # Entrenamiento del modelo
    params_finales, min_vals, range_vals = logistic_regression(params, X_train, y_train, 50000, learning_rate)

    # Evaluación final del modelo en el conjunto de prueba
    evaluate_model(params_finales, X_test, y_test, min_vals, range_vals)

    # Evaluación del modelo en el conjunto de validación
    evaluate_model(params_finales, X_val, y_val, min_vals, range_vals)
