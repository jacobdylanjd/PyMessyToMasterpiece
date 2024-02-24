# Install third-party packages:
import logging
import pandas as pd

# Install project packages:
from ..ml_source.feature_engineering import drop_columns, one_hot_encode_column


def run_feature_engineering(logger: logging.Logger,
                            data: dict) -> pd.DataFrame:
    """
    Run feature engineering stages.

    Args:
        logger (logging.Logger): Logger for logging.
        data (dict): Un-processed dataframe.
    Return:
        df (pd.DataFrame): Processed dataframe.
    """

    logger.info("Run feature engineering")
    df = data['titanic_data']

    logger.info("Drop columns")
    df = drop_columns(df)

    logger.info("Drop NA")
    df = df.dropna()

    logger.info("Run one-hot-encoding")
    df = one_hot_encode_column(df, 'Sex')

    logger.info("Feature engineering complete")

    return df

