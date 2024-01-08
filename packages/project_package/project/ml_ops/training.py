# Install third-party packages:
import mlflow.sklearn
import pandas as pd
import logging

# Install project packages:
from ..ml_source.training import split_data, model_training, log_metadata
from .mlflow import publish_model


def run_training(logger: logging.Logger,
                 config: dict,
                 df: pd.DataFrame) -> pd.DataFrame:
    """
    Split data into train and test datasets.
    Args:
        logger (logging.Logger): Logger for logging.
        config (dict): Project configuration file.
        df (pd.DataFrame): Processed data for model training.
    Return:
        metadata (pd.DataFrame): Trained model metadata.
    """

    logger.info("Run training")

    logger.info("Split data")
    data = split_data(df)

    logger.info("Run model training")
    mlflow.sklearn.autolog(silent=True)
    out_dict = model_training(data)
    mlflow.log_metric("accuracy", out_dict['accuracy'])
    mlflow.log_metric("precision", out_dict['precision'])
    mlflow.log_metric("recall", out_dict['recall'])

    logger.info("Log model metadata")
    metadata = log_metadata(config, out_dict)
    logger.info(f"Added {len(metadata.index)} rows to metadata file")

    logger.info("Publish model to mlflow registry")
    publish_model(logger,
                  config,
                  'production',
                  'model description...')

    logger.info("Training complete")

    return metadata
