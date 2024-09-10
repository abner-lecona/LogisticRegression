# Abner Maximiliano Lecona Nieves
# A01753179

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Lectura del archivo
data = pd.read_csv("500hits.csv", encoding='latin-1')

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

# Escalar los datos de train y test (Min Max)
scaler = MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
x_test_scaled = pd.DataFrame(scaler.fit_transform(x_test),columns=x_test.columns)

#RandomForest con datos escalados
forest_class = RandomForestClassifier(criterion='gini', n_estimators=20, max_depth=7)
forest_class.fit(x_train_scaled, y_train) # Entrenamiento del modelo

# Entrenar el modelo
y_pred_minmax = forest_class.predict(x_test_scaled) 

# Evaluar el rendimiento en el conjunto de entrenamiento
train_score = forest_class.score(x_train_scaled, y_train)
# Evaluar el rendimiento en el conjunto de prueba
test_score = forest_class.score(x_test_scaled, y_test)
# Hacemos la matris de confusion
conf_matrix = confusion_matrix(y_test, y_pred_minmax)

# Imprimir el rendimiento del modelo con los datos escalados
print(f"\nDatos del modelo con datos escalados\n")
print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")

print(f"{classification_report(y_test, y_pred_minmax)}\n")

print("Matriz de Confusión:")
print(conf_matrix)

#RandomForest con datos sin escalar
forest_class = RandomForestClassifier(criterion='gini', n_estimators=20, max_depth=7)
forest_class.fit(x_train, y_train)

# Entrenamiento del modelo
y_pred =forest_class.predict(x_test)

# Evaluar el rendimiento en el conjunto de entrenamiento
train_score = forest_class.score(x_train, y_train)
# Evaluar el rendimiento en el conjunto de prueba
test_score = forest_class.score(x_test, y_test)
# Hacemos la matris de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nDatos del modelo con datos sin escalar\n")
print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")

print(f"{classification_report(y_test, y_pred)}\n")

print("Matriz de Confusión:")
print(conf_matrix)