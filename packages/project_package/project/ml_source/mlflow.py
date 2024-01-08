# Install third-party packages:
import time
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


def wait_until_model_available(model_name: str,
                               model_version: str) -> None:
    """
    Wait until model available in mlflow model registry.
    Args:
        model_name (str): Model name in model registry.
        model_version (dict): Model version most recently published.
    Return:
        None
    """

    client = MlflowClient()

    for _ in range(10):
        status = ModelVersionStatus.from_string(client.get_model_version(name=model_name, version=model_version).status)

        if status == ModelVersionStatus.READY:
            break

        time.sleep(1)

    return None
