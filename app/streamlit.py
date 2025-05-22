import streamlit as st
import pandas as pd
import pickle
from joblib import dump, load
import io
import base64

# === Contraseña simple ===
PASSWORD_CORRECTA = "Meli"


st.image("mercado-libre-logo-1.png", width=800)


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


# Carga del modelo .pkl
@st.cache_resource
def cargar_modelo():
    return load('../Models/best_model_production.pkl')

modelo = cargar_modelo()

# Subida del archivo CSV
archivo_csv = st.file_uploader("Sube tu archivo CSV", type="csv")

# Inicializamos el DataFrame
df = None

# Si se subió un archivo
if archivo_csv is not None:
    df = pd.read_csv(archivo_csv)
    st.write("Vista previa del archivo:")
    st.dataframe(df)

    # Botón para hacer predicciones
    if st.button("Predecir"):
        try:
            # Realiza predicciones (ajusta esto según el input que espera tu modelo)
            predicciones = modelo.predict(df)

            # Agrega la columna de predicciones
            df['prediccion'] = predicciones

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