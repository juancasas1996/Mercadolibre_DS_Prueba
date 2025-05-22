import mlflow
from mlflow.tracking import MlflowClient


# === CONFIGURACIÓN ===
mlflow.set_tracking_uri("file:./Experiments")
EXPERIMENT_NAME = "XGBoost_Experiment"
MODEL_NAME = "Best_Production_Model"
RUN_NAME = "XGBoost"


# === CLIENTE ===
client = MlflowClient()


experiments = mlflow.search_experiments()
print("Experiments disponibles:")
for exp in experiments:
    print(f"- {exp.name}")




# === Obtener ID del experimento ===
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"No se encontró el experimento '{EXPERIMENT_NAME}'")

# === Buscar runs y elegir el mejor (por AUC, por ejemplo) ===
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    order_by=["metrics.roc_auc DESC"]
)

if not runs:
    raise ValueError(f"No se encontraron runs con nombre '{RUN_NAME}'")

best_run = runs[0]
run_id = best_run.info.run_id
print(f"✅ Mejor run encontrado: {run_id}")
print(f"AUC: {best_run.data.metrics['roc_auc']:.4f}")
print(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}")

# === Registrar modelo ===
model_uri = f"runs:/{run_id}/xgboost_pipeline_v1"
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"📦 Modelo registrado: {MODEL_NAME} (versión {result.version})")

# === Promover modelo a producción ===
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)
print(f"🚀 Modelo promovido a producción ✅")