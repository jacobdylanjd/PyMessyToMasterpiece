import pandas as pd
import logging
from modules.ml_source.training import split_data, model_training, log_metadata, save_model


def run_training(logger: logging.Logger,
                 df: pd.DataFrame) -> pd.DataFrame:
    """Split data into train and test datasets.

        Args:
            logger (logging.Logger): Logger for logging.
            df (pd.DataFrame): Processed data for model training.
        Return:
            metadata (pd.DataFrame): Trained model metadata.
    """

    logger.info("Run training")

    logger.info("Split data")
    data = split_data(df)

    logger.info("Run model training")
    out_dict = model_training(data)

    logger.info("Log model metadata")
    metadata = log_metadata(out_dict)
    logger.info(f"Added {len(metadata.index)} rows to metadata file")

    logger.info("Save model")
    save_model(out_dict['model'])

    logger.info("Training complete")

    return metadata

