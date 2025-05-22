Mercado Libre - Clasificación de Productos Nuevos vs Usados

Este proyecto busca predecir si un producto listado en Mercado Libre es nuevo o usado, aplicando un pipeline completo de ciencia de datos: exploración, limpieza, ingeniería de variables, modelado, evaluación y despliegue.

⸻

1. Dataset

El conjunto de datos original contiene descripciones, información de pagos, métodos de pago, precios, atributos, entre otros campos, muchos de ellos anidados en listas o diccionarios.

⸻

2. Exploración y Limpieza (Notebook: 01_EDA.ipynb)

Tratamiento de Columnas Complejas (listas/diccionarios)
	•	descriptions: se extrajo description_id.
	•	non_mercado_pago_payment_methods: se extrajo el tipo de pago principal. Se analizó la distribución de tipos de tarjeta y se agruparon todas en un solo grupo: “Tarjeta”.

Columnas tipo lista:

Se analizaron: sub_status, variations, deal_ids, attributes, coverage_areas, shipping.methods, shipping.tags. Se definió acción según % de vacíos y su relación con la variable objetivo (condition). Se generaron variables binarias como has_variations, has_attributes, y se eliminaron otras.

Columna tags:
	•	Se encontró un 75% de filas con valores.
	•	Se aplicó OneHotEncoding tras binarización con MultiLabelBinarizer.

Valores nulos:
	•	Se estandarizó None, none, '' a np.nan.
	•	Se eliminaron columnas con >95% de nulos y sin información relevante.

Columnas constantes:
	•	Se identificaron y eliminaron columnas con un único valor no nulo.

Columnas con valores únicos en 95%+
	•	Se eliminaron columnas como title, thumbnail, secure_thumbnail, permalink, etc.

Columna warranty (60% vacía):
	•	Se aplicó NLP + Clustering:
	•	Se usó el modelo sentence-transformers/all-MiniLM-L12-v2 para generar embeddings.
	•	Clustering con KMeans(n_clusters=4).
	•	Se creó la variable has_warranty usando el texto y reglas adicionales.

Fechas y tiempos:
	•	De date_created se extrajo: year, month, day, weekday, is_weekend.
	•	De start_time y stop_time se extrajo la variable active_period.

Agrupaciones y simplificaciones:
	•	currency_id → binaria Currency_ARS.
	•	status → “active”, “paused”, “other”.
	•	shipping.mode → agrupación “me1” → “custom”.
	•	seller_address.state.name y state.id eran duplicadas → se eliminó una.

Correlaciones numéricas:
	•	Se encontró alta correlación entre base_price y price, y entre initial_quantity y available_quantity. Se eliminaron las columnas redundantes.

⸻

3. Modelos (Notebooks: 03_, 04_, 05_)

Modelos entrenados:
	1.	XGBoost
	•	Búsqueda de hiperparámetros con RandomizedSearchCV
	•	Variables categóricas codificadas con OneHotEncoder (baja cardinalidad) y OrdinalEncoder (alta cardinalidad)
	•	Mejores resultados:
	•	Accuracy: 0.9021
	•	ROC AUC: 0.9663
	2.	Logistic Regression
	•	GridSearch con regularización y codificación de variables.
	•	Resultado:
	•	Accuracy: 0.8467
	•	ROC AUC: 0.9207
	3.	Neural Network (MLPClassifier)
	•	Se aplicó reducción del dataset por limitaciones de memoria.
	•	Resultado:
	•	Accuracy: 0.8625
	•	ROC AUC: 0.9358

⸻

4. Tracking con MLflow
	•	Todos los experimentos fueron registrados usando MLflow.
	•	Se guardaron hiperparámetros, métricas, modelos y artefactos (curvas ROC, learning curves).
	•	El mejor modelo fue seleccionado automáticamente según accuracy y ROC AUC.

⸻

5. Producción - App Streamlit
	•	Se construyó una interfaz con Streamlit (App/streamlit.py) protegida por contraseña (Meli).
	•	El usuario puede subir un CSV con el mismo formato de X_test.
	•	La app devuelve un archivo descargable con las predicciones.

⸻

6. Estructura de Carpetas

MercadoLibre_Test/
├── Data/
│   ├── bronze/
│   └── Gold/
│
├── Notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_XGBoost.ipynb
│   ├── 04_Model_Logistic.ipynb
│   ├── 05_Neural_Network.ipynb
│   ├── 06_Compare.ipynb
│
├── Models/
│   └── best_model_production.pkl
│
├── Experiments/
│   └── (MLflow tracking runs)
│
├── App/
│   ├── streamlit.py
│   ├── Data_Processing.py
│   ├── Register_Best_Model.py
│
├── Artefacts/
│   └── preprocessors/
│       ├── kmeans_warranty.pkl
│       ├── mlb_tags.pkl
│       ├── warranty_cluster_map.pkl
│
└── Docker/ *(En desarrollo)*
    ├── Dockerfile
    ├── app.py
    └── requirements.txt


⸻

7. Pendientes
	•	Finalizar Docker para permitir despliegue reproducible de la app.
	•	Extender la validación cruzada para modelos complejos (si hay más recursos).

⸻

8. Requisitos
	•	Python 3.11
	•	scikit-learn, xgboost, streamlit, mlflow, sentence-transformers, pandas, joblib, matplotlib

Instalación:

pip install -r Docker/requirements.txt


⸻

9. Autor

Ejercicio para Mercado Libre - Desarrollado por [Juan Sebastian Casas Castillo]
