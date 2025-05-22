import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from joblib import dump
import os

def seleccionar_y_guardar_mejor_modelo(output_path="../Models/best_model_production.pkl"):
    """
    Busca el modelo con mejor accuracy registrado en MLflow
    y lo guarda en el path especificado.
    """
    mlflow.set_tracking_uri("file:../Experiments")
    client = MlflowClient()

    # Obtener todos los experimentos y sus runs finalizados
    experiments = mlflow.search_experiments()
    all_runs = []

    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'"
        )
        for run in runs:
            all_runs.append({
                "run_id": run.info.run_id,
                "experiment_name": exp.name,
                "run_name": run.data.tags.get("mlflow.runName", None),
                "accuracy": run.data.metrics.get("accuracy", None),
                "roc_auc": run.data.metrics.get("roc_auc", None)
            })

    df = pd.DataFrame(all_runs)
    df_filtered = df.dropna(subset=["accuracy"])
    df_sorted = df_filtered.sort_values(by="accuracy", ascending=False)

    if df_sorted.empty:
        print("❌ No se encontró ningún modelo con métricas registradas.")
        return

    # Mejor run
    best_run_id = df_sorted.iloc[0]["run_id"]

    try:
        # Detectar artefactos del run
        artifacts = client.list_artifacts(best_run_id)
        model_dirs = [a.path for a in artifacts if a.is_dir]

        if not model_dirs:
            print("❌ No se encontró modelo registrado en el mejor run.")
            return

        # Cargar y guardar
        model_uri = f"runs:/{best_run_id}/{model_dirs[0]}"
        best_model = mlflow.sklearn.load_model(model_uri)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dump(best_model, output_path)
        print(f"✅ Mejor modelo guardado en {output_path}")

    except Exception as e:
        print(f"❌ Error al cargar o guardar el modelo: {e}")