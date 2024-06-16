from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
import tensorflow_hub as hub
from keras.utils import get_custom_objects
from keras.preprocessing import image
import os
from flask_cors import CORS
import logging
from PIL import Image
import cv2
import csv
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

get_custom_objects().update({'KerasLayer': hub.KerasLayer})
#modelo = keras.models.load_model(r"C:\Users\zloop\Downloads\PRY20232084.h5")
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del cuerpo de la solicitud
    request_data = request.json

    # Asegurarse de que los datos sean válidos y tengan todas las claves necesarias
    if request_data is None or "tipo_de_materia_prima" not in request_data or "cantidad" not in request_data \
            or "mes_de_consumo" not in request_data or "año" not in request_data \
            or "compra_siguiente_mes" not in request_data:
        return jsonify({'error': 'Datos de solicitud incompletos o incorrectos'}), 400

    nuevos_datos = request_data

    # Ruta del archivo CSV existente
    archivo_csv = r"PRY20232084 Dataset.csv"

    # Leer el archivo CSV
    data = pd.read_csv(archivo_csv, encoding='utf-8-sig')

    # Verificar si hay un registro existente para la materia prima del mismo mes y año
    filtro = (data['tipo_de_materia_prima'] == nuevos_datos["tipo_de_materia_prima"]) & \
             (data['mes_de_consumo'] == nuevos_datos["mes_de_consumo"]) & \
             (data['año'] == nuevos_datos["año"])

    registros_existente = data.loc[filtro]

    if not registros_existente.empty:
        # Actualizar los datos existentes
        data.loc[filtro, 'consumo_mensual'] = nuevos_datos["cantidad"]
        data.loc[filtro, 'compra_siguiente_mes'] = nuevos_datos["compra_siguiente_mes"]
    else:
        # Agregar los nuevos datos al DataFrame
        nuevos_registro = pd.DataFrame([nuevos_datos])
        data = pd.concat([data, nuevos_registro], ignore_index=True)

    # Escribir el DataFrame actualizado de nuevo al archivo CSV
    data.to_csv(archivo_csv, index=False, encoding='utf-8-sig')

    # Codifica las variables categóricas
    X = pd.get_dummies(data, columns=['tipo_de_materia_prima', 'mes_de_consumo', 'año'])
    y = data['compra_siguiente_mes']

    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escala los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define el modelo
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compila el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrena el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
     
    # Predicciones con las mismas características y formato
    X_predicciones = pd.get_dummies(data, columns=['tipo_de_materia_prima', 'mes_de_consumo', 'año'])
    X_predicciones = scaler.transform(X_predicciones)

    predictions = model.predict(X_predicciones)

    # Unir datos originales con predicciones
    data['prediccion'] = predictions

    # Obtener los últimos 6 meses de datos reales y predicciones para el tipo de materia prima proporcionado como entrada
    resultados_tipo_materia_prima = data[data['tipo_de_materia_prima'] == request_data["tipo_de_materia_prima"]].tail(6)
    predicciones = resultados_tipo_materia_prima['prediccion'].tolist()
    reales = resultados_tipo_materia_prima['compra_siguiente_mes'].tolist()

    # Crear el diccionario de resultados
    results = {
        'Name': request_data["tipo_de_materia_prima"],
        'Prediction': predicciones,
        'Real': reales
    }

    return jsonify({'results': results})

@app.route('/last_six_months', methods=['POST'])
def last_six_months():
    # Obtener los datos del cuerpo de la solicitud
    request_data = request.json

    # Asegurarse de que los datos sean válidos y tengan todas las claves necesarias
    if request_data is None or "tipo_de_materia_prima" not in request_data or "año" not in request_data:
        return jsonify({'error': 'Datos de solicitud incompletos o incorrectos'}), 400

    # Ruta del archivo CSV existente
    archivo_csv = r"PRY20232084 Dataset.csv"

    # Lee el archivo CSV
    data = pd.read_csv(archivo_csv)

    # Codifica las variables categóricas
    X = pd.get_dummies(data, columns=['tipo_de_materia_prima', 'mes_de_consumo', 'año'])
    y = data['compra_siguiente_mes']

    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escala los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define el modelo
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compila el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrena el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
     
    # Predicciones con las mismas características y formato
    X_predicciones = pd.get_dummies(data, columns=['tipo_de_materia_prima', 'mes_de_consumo', 'año'])
    X_predicciones = scaler.transform(X_predicciones)

    predictions = model.predict(X_predicciones)

    # Unir datos originales con predicciones
    data['prediccion'] = predictions

    # Obtener los últimos 6 meses de datos reales y predicciones para el tipo de materia prima proporcionado como entrada
    resultados_tipo_materia_prima = data[data['tipo_de_materia_prima'] == request_data["tipo_de_materia_prima"]].tail(7)
    predicciones = resultados_tipo_materia_prima['prediccion'].tolist()
    reales = resultados_tipo_materia_prima['compra_siguiente_mes'].tolist()

    # Crear el diccionario de resultados
    results = {
        'Name': request_data["tipo_de_materia_prima"],
        'Prediction': predicciones,
        'Real': reales[:-1]
    }

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081,debug=True)