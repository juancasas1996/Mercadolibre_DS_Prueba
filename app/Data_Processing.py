import pandas as pd
import os
import ast
import pandas as pd
import numpy as np
import ast
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer




def preprocess(df, is_train=True, mlb_tags=None, model=None, kmeans=None, cluster_map=None):
    df = df.copy()

    df.drop(columns=['pictures'], inplace=True, errors='ignore')

    # === 1. Parsear 'descriptions'
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

    # === 2. Método de pago principal
    def extract_primary_payment_method(lst):
        if isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], dict):
            return lst[0].get('description', 'unknown')
        return 'unknown'
    df['payment_method'] = df['non_mercado_pago_payment_methods'].apply(extract_primary_payment_method)
    df.drop(columns=['non_mercado_pago_payment_methods'], inplace=True)

    # === 3. Flags binarias de listas
    df['has_variations'] = df['variations'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df['has_deal_ids'] = df['deal_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df['has_attributes'] = df['attributes'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df.drop(columns=['sub_status', 'coverage_areas', 'variations', 'deal_ids', 'attributes'], inplace=True)

    # === 4. Tags → one-hot
    tags_list = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    if is_train:
        mlb_tags = MultiLabelBinarizer()
        tags_encoded = pd.DataFrame(mlb_tags.fit_transform(tags_list), columns=[f'tag_{t}' for t in mlb_tags.classes_], index=df.index)
    else:
        tags_encoded = pd.DataFrame(mlb_tags.transform(tags_list), columns=[f'tag_{t}' for t in mlb_tags.classes_], index=df.index)
    df = pd.concat([df, tags_encoded], axis=1)
    df.drop(columns=['tags'], inplace=True)

    # === 5. Nulos
    df.replace(['none', 'None', ''], np.nan, inplace=True)
    df = df.applymap(lambda x: np.nan if x is None else x)

    # === 6. Eliminar columnas innecesarias
    drop_cols = [
        'listing_source', 'international_delivery_mode', 'official_store_id', 'differential_pricing', 'original_price',
        'video_id', 'catalog_product_id', 'subtitle', 'shipping.dimensions', 'shipping.free_methods',
        'shipping.methods', 'shipping.tags', 'site_id', 'seller_address.country.name', 'seller_address.country.id',
        'last_updated', 'id', 'thumbnail', 'title', 'secure_thumbnail', 'permalink', 'description_id',
        'seller_address.city.id', 'parent_item_id', 'seller_address.state.id'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # === 7. Bools a float
    bool_cols = ['accepts_mercadopago', 'automatic_relist', 'shipping.local_pick_up', 'shipping.free_shipping', 'has_variations', 'has_deal_ids', 'has_attributes']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # === 8. Garantía → embeddings → cluster → has_warranty
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

    # === 9. Fechas
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

    # === 10. Currency/status/shipping.mode
    df["Currency_ARS"] = (df["currency_id"] == "ARS").astype(int)
    df.drop(columns=["currency_id"], inplace=True)

    df["status"] = df["status"].apply(lambda x: x if x in ["active", "paused"] else "other")
    df["shipping.mode"] = df["shipping.mode"].replace("me1", "custom")


    # Agrupación de métodos de pago
    tarjetas = [
        "Tarjeta de crédito", "MasterCard", "Mastercard Maestro",
        "Visa", "Visa Electron"
    ]

    df["payment_method_grouped"] = df["payment_method"]
    df.loc[df["payment_method"].isin(tarjetas), "payment_method_grouped"] = "Tarjeta"
    df.loc[df["payment_method"] == "Transferencia bancaria", "payment_method_grouped"] = "Transferencia"
    df.loc[df["payment_method"].isin(["Efectivo", "Giro postal", "Cheque certificado"]), "payment_method_grouped"] = "Efectivo"
    df.loc[df["payment_method"].isin(["Acordar con el comprador", "Contra reembolso"]), "payment_method_grouped"] = "A convenir"
    df["payment_method_grouped"] = df["payment_method_grouped"].fillna("unknown")
    df.drop(columns=["payment_method"], inplace=True)

    # Se eliminan por alta correlacion entre columnas
    df.drop(columns=['base_price', 'initial_quantity'], inplace=True)


    if is_train:

        os.makedirs("../artifacts/preprocessors", exist_ok=True)

        joblib.dump(kmeans, "../artifacts/preprocessors/kmeans_warranty.pkl")
        joblib.dump(cluster_map, "../artifacts/preprocessors/warranty_cluster_map.pkl")
        joblib.dump(mlb_tags, "../artifacts/preprocessors/mlb_tags.pkl")

    return df


