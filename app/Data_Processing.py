import os
import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def preprocess(df, is_train=True, mlb_tags=None, model=None, kmeans=None, cluster_map=None):
    df = df.copy()

    df.drop(columns=['pictures'], inplace=True)
    
    # === 1. Parsear descripción
    def parse_description(cell):
        if isinstance(cell, list) and len(cell) > 0:
            try:
                return ast.literal_eval(cell[0])
            except:
                return None
        return None

    df['descriptions'] = df['descriptions'].apply(parse_description)
    df['description_id'] = df['descriptions'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
    df.drop(columns=['descriptions'], inplace=True)

    # === 2. Procesar métodos de pago
    df['payment_descriptions'] = df['non_mercado_pago_payment_methods'].apply(
        lambda lst: [d['description'] for d in lst] if isinstance(lst, list) else []
    )
    mlb = MultiLabelBinarizer()
    payment_dummies = pd.DataFrame(
        mlb.fit_transform(df['payment_descriptions']),
        columns=[f"non_mercado_pago_payment_methods_description_{desc}" for desc in mlb.classes_],
        index=df.index
    )
    df = pd.concat([df, payment_dummies], axis=1)

    # Agrupar métodos de pago relacionados con tarjeta
    tarjetas = ["Visa", "MasterCard", "Visa Electron", "Mastercard Maestro", "American Express", "Diners", "Tarjeta de crédito"]
    tarjetas_cols = [
        f"non_mercado_pago_payment_methods_description_{t}"
        for t in tarjetas if f"non_mercado_pago_payment_methods_description_{t}" in df.columns
    ]
    df["payment_method_Tarjeta"] = df[tarjetas_cols].sum(axis=1).clip(upper=1)
    df.drop(columns=['non_mercado_pago_payment_methods'] + tarjetas_cols + ['payment_descriptions'], inplace=True)

    # === 3. Columnas binarias a partir de listas
    df['has_variations'] = df['variations'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df['has_attributes'] = df['attributes'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df.drop(columns=['sub_status', 'coverage_areas', 'variations', 'deal_ids', 'attributes', 'shipping.methods', 'shipping.tags'], inplace=True)

    # === 4. Tags
    tags_list = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    if is_train:
        mlb_tags = MultiLabelBinarizer()
        tags_encoded = pd.DataFrame(mlb_tags.fit_transform(tags_list), columns=[f'tag_{t}' for t in mlb_tags.classes_], index=df.index)
    else:
        tags_encoded = pd.DataFrame(mlb_tags.transform(tags_list), columns=[f'tag_{t}' for t in mlb_tags.classes_], index=df.index)
    df = pd.concat([df, tags_encoded], axis=1)
    df.drop(columns=['tags'], inplace=True)

    # === 5. Limpieza general de nulos y strings vacíos
    df.replace(['none', 'None', ''], np.nan, inplace=True)
    df = df.applymap(lambda x: np.nan if x is None else x)

    # === 6. Eliminar columnas con demasiados nulos o con correlaciones altas con otras variables 
    drop_cols = [
        'listing_source', 'international_delivery_mode', 'official_store_id', 'differential_pricing', 'original_price',
        'video_id', 'catalog_product_id', 'subtitle', 'shipping.dimensions', 'shipping.free_methods',
        'last_updated', 'id', 'thumbnail', 'title', 'secure_thumbnail', 'permalink',
        'seller_address.city.id', 'parent_item_id', 'seller_address.state.id', 'base_price', 'initial_quantity', 'site_id', 'seller_address.country.name', 'seller_address.country.id', 'description_id'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # === 7. Conversión de booleanos
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    df[bool_cols] = df[bool_cols].astype(float)

    # === 8. Procesamiento de 'warranty' con embeddings
    if is_train:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        texts = df['warranty'].dropna().unique()
        embeddings = model.encode(texts, show_progress_bar=False)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        cluster_map = dict(zip(texts, clusters))

    df['warranty_cluster'] = df['warranty'].map(cluster_map).fillna(-1)
    mapeo_final = {0: "tiene_garantia", 1: "tiene_garantia", 2: "tiene_garantia", 3: "no_menciona", -1: "no_menciona"}
    df['warranty_label'] = df['warranty_cluster'].map(mapeo_final)

    def corregir_warranty(texto, etiqueta_inicial):
        if pd.isna(texto):
            return etiqueta_inicial
        texto_limpio = texto.lower()
        if "sin garantia" in texto_limpio or "sin garantía" in texto_limpio:
            return "sin_garantia"
        return etiqueta_inicial

    df['warranty_label'] = df.apply(lambda row: corregir_warranty(row['warranty'], row['warranty_label']), axis=1)
    df['has_warranty'] = (df['warranty_label'] == 'tiene_garantia').astype(float)
    df.drop(columns=['warranty', 'warranty_cluster', 'warranty_label'], inplace=True)

    # === 9. Procesamiento de fechas
    df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
    df['year'] = df['date_created'].dt.year
    df['month'] = df['date_created'].dt.month
    df['day'] = df['date_created'].dt.day
    df['weekday'] = df['date_created'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df.drop(columns=['date_created'], inplace=True)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", unit="ms")
    df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce", unit="ms")
    df["active_period"] = (df['stop_time'] - df['start_time']).dt.days
    df.drop(columns=["start_time", "stop_time"], inplace=True)


    # === 10. Ajustes finales
    df["Currency_ARS"] = (df["currency_id"] == "ARS").astype(int)
    df.drop(columns=["currency_id"], inplace=True)

    df["status"] = df["status"].apply(lambda x: x if x in ["active", "paused"] else "other")
    df["shipping.mode"] = df["shipping.mode"].replace("me1", "custom")

    # === 11. Eliminar filas con NaN finales
    if is_train:
        df = df.dropna(axis=0)

    # === 12. Guardar artefactos si es entrenamiento
    if is_train:
        os.makedirs("../artifacts/preprocessors", exist_ok=True)
        joblib.dump(kmeans, "../artifacts/preprocessors/kmeans_warranty.pkl")
        joblib.dump(cluster_map, "../artifacts/preprocessors/warranty_cluster_map.pkl")
        joblib.dump(mlb_tags, "../artifacts/preprocessors/mlb_tags.pkl")

    # Para test, cargando los modelos previamente guardados

    return df


def procesar_test(X_test):
    # Para test, cargando los modelos previamente guardados
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    kmeans = joblib.load("../artifacts/preprocessors/kmeans_warranty.pkl")
    cluster_map = joblib.load("../artifacts/preprocessors/warranty_cluster_map.pkl")
    mlb_tags = joblib.load("../artifacts/preprocessors/mlb_tags.pkl")

    df = preprocess(X_test, is_train=False, model=model, kmeans=kmeans, cluster_map=cluster_map, mlb_tags=mlb_tags)

    return df