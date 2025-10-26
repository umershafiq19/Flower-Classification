import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/umershafiq19/mlflow-flask-flowers.mlflow")
mlflow.set_registry_uri("https://dagshub.com/umershafiq19/mlflow-flask-flowers.mlflow")

client = MlflowClient()

client.transition_model_version_stage(
    name="flowers_classifier",
    version=1,
    stage="Production"
)

