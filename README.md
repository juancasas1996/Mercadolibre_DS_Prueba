# 🛒 Mercado Libre - Clasificación de Productos Nuevos vs Usados

Este proyecto busca predecir si un producto listado en Mercado Libre es **nuevo** o **usado**, implementando un pipeline completo de ciencia de datos: exploración, limpieza, ingeniería de variables, modelado, evaluación y despliegue.

---

## 📁 1. Dataset

El dataset contiene información detallada de publicaciones, incluyendo:
- Descripciones y garantías
- Métodos de pago
- Atributos y precios
- Fechas, ubicaciones y más

Muchos campos vienen anidados como **listas** o **diccionarios**, lo que motivó un procesamiento profundo.

---

## 🔍 2. Exploración y Limpieza (`01_EDA.ipynb`)

### 🔹 Columnas Complejas (listas/diccionarios)
- `descriptions`: Se extrajo `description_id`.
- `non_mercado_pago_payment_methods`: Se extrajo el método de pago principal. Agrupación de métodos relacionados con tarjetas bajo “Tarjeta”.

### 🔹 Columnas tipo lista
Columnas como `sub_status`, `variations`, `deal_ids`, `attributes`, etc., fueron evaluadas según su % de vacíos y su señal predictiva (`condition`). Resultado:
- Se eliminaron columnas con demasiados nulos o sin señal.
- Se crearon variables binarias como `has_variations`, `has_attributes`.

### 🔹 Columna `tags`
- 75.10% de filas contenían tags útiles.
- Se aplicó `MultiLabelBinarizer` + `OneHotEncoding`.

### 🔹 Valores nulos
- Estandarización de `None`, `none`, `''` a `np.nan`.
- Se eliminaron columnas con más del 95% de nulos si no tenían valor predictivo.

### 🔹 Columnas constantes y casi constantes
- Se eliminaron columnas con un solo valor (ej: `site_id`, `thumbnail`, etc.).
- También columnas con 95% del mismo valor.

### 🔹 Columna `warranty` (60% vacía)
- Se aplicó **NLP + Clustering**:
  - Embeddings con `sentence-transformers/all-MiniLM-L12-v2`.
  - Clustering con `KMeans(n_clusters=4)`.
  - Se creó una variable binaria: `has_warranty`.

### 🔹 Fechas y tiempos
- De `date_created`: se extrajeron `year`, `month`, `day`, `is_weekend`, etc.
- De `start_time` y `stop_time`: se generó `active_period`.

### 🔹 Agrupación y simplificación de categorías
- `currency_id` → `Currency_ARS` (binaria)
- `status` → agrupado como `active`, `paused`, `other`
- `shipping.mode` → agrupación personalizada

### 🔹 Correlaciones fuertes
- `price` ≈ `base_price` → se eliminó uno
- `initial_quantity` ≈ `available_quantity` → se eliminó uno
- `seller_address.state.id` y `.name` eran equivalentes → se eliminó uno

---

## 🤖 3. Modelos Entrenados

### 📘 `03_Model_XGBoost.ipynb`
Entrenamiento de un modelo **XGBoost** con:
- Codificación mixta:
  - `OrdinalEncoder` para columnas de alta cardinalidad (`seller_id`, `city.name`, `category_id`)
  - `OneHotEncoder` para el resto de columnas categóricas
  - `StandardScaler` para numéricas
- Búsqueda con `RandomizedSearchCV`, 20 combinaciones probadas, validación con `StratifiedKFold`.
- Registro de métricas (`accuracy`, `roc_auc`) y curva ROC en MLflow.
- Modelo registrado con `input_example` y `signature`.

---

### 📘 `04_Model_Logistic.ipynb`
Entrenamiento de **regresión logística** con:
- Transformador personalizado `TopCityTransformer` para agrupar ciudades frecuentes.
- Pipeline con detección automática de columnas categóricas y numéricas (`make_column_selector`).
- Submuestreo del dataset para acelerar entrenamiento.
- Optimización de hiperparámetros (`C`, `solver`) vía `RandomizedSearchCV`.
- Registro completo del modelo y métricas en MLflow.

---

### 📘 `05_Neural_Network.ipynb`
Entrenamiento de una **red neuronal simple (MLPClassifier)** con:
- Agrupación de ciudades (`city_grouped`) y eliminación de columnas de alta cardinalidad.
- Preprocesamiento tradicional con `OneHotEncoder` + `StandardScaler`.
- Arquitectura: `hidden_layer_sizes=(64,)`, `max_iter=200`.
- Registro directo del modelo sin tuning de hiperparámetros.

---

### 📘 `06_Compare.ipynb`
- Recolecta todos los experimentos registrados en **MLflow**.
- Compara métricas (`accuracy`, `roc_auc`) entre modelos.
- Carga cada modelo y genera su **matriz de confusión** y `classification_report`.
- Selecciona automáticamente el mejor modelo (según `accuracy`) y lo guarda como `best_model_production.pkl`.

---

### 📈 Resumen de resultados

| Modelo               | Accuracy | ROC AUC |
|----------------------|----------|---------|
| **XGBoost**          | 0.9021   | 0.9663  |
| Neural Network (MLP) | 0.8625   | 0.9358  |
| Logistic Regression  | 0.8467   | 0.9207  |

---

## 📊 4. Tracking de Experimentos

Todos los modelos y sus ejecuciones fueron registrados con **MLflow**, incluyendo:
- Hiperparámetros evaluados
- Métricas (`accuracy`, `roc_auc`)
- Artefactos como curvas ROC
- Modelos entrenados en cada run

El mejor modelo es seleccionado automáticamente con base en su métrica.

---

## 🚀 5. Despliegue con Streamlit

Una aplicación desarrollada con **Streamlit** permite:
- Subir archivos `.jsonlines` similares a `X_test` raw
- Procesar automáticamente los datos y obtener predicciones
- Descargar los resultados en `.csv`
- Acceso protegido con contraseña: `Meli`

---

## 🗂️ 6. Estructura del Proyecto

<pre lang="markdown"><code>

MercadoLibre_Test/
├── Data/
│   ├── bronze/
│   └── gold/
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
└── Docker/ (En desarrollo)
    ├── Dockerfile
    ├── app.py
    └── requirements.txt

</code></pre>

---

## ⏳ 7. Pendientes

- Finalizar el contenedor Docker para facilitar el despliegue reproducible.
- Optimizar el entrenamiento de modelos más pesados si se dispone de mayor capacidad computacional.

---

## ⚙️ 8. Requisitos

- Python 3.11+
- `scikit-learn`, `xgboost`, `streamlit`, `mlflow`, `sentence-transformers`, `pandas`, `matplotlib`, `joblib`

Instalación rápida:

```bash
pip install -r Docker/requirements.txt