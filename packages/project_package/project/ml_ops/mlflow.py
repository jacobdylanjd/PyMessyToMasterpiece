# Install third-party packages:
import mlflow
import logging
from mlflow.tracking.client import MlflowClient

# Install project packages:
from ..ml_source.mlflow import wait_until_model_available


def publish_model(logger: logging.Logger,
                  config: dict,
                  stage: str,
                  description: str) -> None:
    """
    Publish model to mlflow model registry.
    Args:
        logger (logging.Logger): Logger for logging.
        config (dict): Project configuration file.
        stage (str): model stage.
        description (str): model description for registry.
    Return:
        None
    """

    client = MlflowClient()
    mlflow_run_id = mlflow.active_run().info.run_id
    model_name = config['output_model_publish_name']

    if config['environment'] == 'local':
        mlflow.register_model(f"runs:/{mlflow_run_id}/model", model_name)
        new_model_version = client.get_latest_versions(model_name)[0].version

        logger.info("Wait for model to be available in model registry")
        wait_until_model_available(model_name, new_model_version)

        logger.info("Update model details in registry")
        client.update_registered_model(
            name=model_name,
            description=description
        )

        logger.info(f"Transition model to {stage} in model registry")
        client.transition_model_version_stage(
            name=model_name,
            version=new_model_version,
            stage=stage,
            archive_existing_versions=True
        )

    else:
        raise ValueError(f"Environment {config['environment']} is not currently supported")

    mlflow.set_tag("stage", stage)
    logger.info(f"Published model: {model_name} version {new_model_version} to stage {stage}")

    return None

