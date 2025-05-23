import streamlit as st
import pandas as pd
import pickle
from joblib import dump, load
import io
from Register_Best_Model import seleccionar_y_guardar_mejor_modelo
import base64

import pandas as pd
from collections import Counter
import sys
import os
import ast
import json
import pandas as pd
import numpy as np
import ast
import joblib
from Data_Processing import preprocess, procesar_test
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler




LOGO_PATH = os.path.join(os.path.dirname(__file__), "mercado-libre-logo-1.png")




pd.set_option('display.max_columns', None)






# === Contraseña simple ===
PASSWORD_CORRECTA = "Meli"


st.image(LOGO_PATH, width=800)


# Input de contraseña
password = st.text_input("Introduce la contraseña para continuar:", type="password")

if password != PASSWORD_CORRECTA:
    st.warning("🔐 Por favor, introduce la contraseña correcta.")
    st.stop()


st.markdown(
    """
    <h1 style='text-align: center; color: #0054A6;'>
        📦 Predicción de Producto Nuevo en Mercado Libre 📦
    </h1>
    """,
    unsafe_allow_html=True
)

# seleccionar_y_guardar_mejor_modelo("../Models/best_model_production.pkl")

path_docker = "Models/best_model_production.pkl"
path_local = "../Models/best_model_production.pkl"
output_path = path_docker if os.path.exists("/app") else path_local

seleccionar_y_guardar_mejor_modelo(output_path)





# === Cargar el modelo
@st.cache_resource
def cargar_modelo():
    path_docker = "/app/Models/best_model_production.pkl"
    path_local = "../Models/best_model_production.pkl"
    model_path = path_docker if os.path.exists(path_docker) else path_local
    return load(model_path)

modelo = cargar_modelo()

# Subida del archivo CSV
archivo_json = st.file_uploader("Sube tu archivo .jsonlines", type=".jsonlines")

# Inicializamos el DataFrame
df = None

# Si se subió un archivo
if archivo_json is not None:

    # Leer cada línea del archivo como JSON
    lineas = archivo_json.readlines()
    json_list = [json.loads(line.decode("utf-8")) for line in lineas]

    df = pd.json_normalize(json_list)
    df = procesar_test(df)

    st.write("Vista previa del archivo:")
    st.dataframe(df)

    # Botón para hacer predicciones
    if st.button("Predecir"):
        try:
            # Realiza predicciones (ajusta esto según el input que espera tu modelo)
            predicciones = modelo.predict(df)

            # Agrega la columna de predicciones
            df['condition'] = predicciones
            df['condition'] = df["condition"].map({0: "used", 1: "new"})

            st.success("¡Predicciones generadas!")
            st.dataframe(df)

            # Botón de descarga
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="Descargar CSV con predicciones",
                data=buffer,
                file_name="predicciones.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error al predecir: {e}")