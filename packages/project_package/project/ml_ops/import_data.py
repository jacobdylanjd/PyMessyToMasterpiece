# Install third-party packages:
import pandas as pd
import logging

# Install project packages:
from ..ml_source.import_data import import_data, run_data_quality_tests


def run_import_data(logger: logging.Logger,
                    config: dict) -> pd.DataFrame:
    """
    Run import data stages.

    Args:
        logger (logging.Logger): Logger for logging.
        config (dict): Project configuration file.
    Return:
        df (pd.DataFrame): Imported data.
    """

    logger.info("Run import data")
    df = import_data(config)

    logger.info("Run data quality tests")
    run_data_quality_tests(df)
    logger.info("Data quality tests passed")

    logger.info("Import data complete")

    return df

