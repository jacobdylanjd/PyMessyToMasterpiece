# Install third-party packages:
import logging

# Install project packages:
from ..ml_source.import_data import import_data, run_data_quality_tests


def run_import_data(logger: logging.Logger,
                    config: dict) -> dict:
    """
    Run import data stages.

    Args:
        logger (logging.Logger): Logger for logging.
        config (dict): Project configuration file.
    Return:
        data (dict): Imported data.
    """

    imported_data = {}

    logger.info("Run import data")
    data = import_data(config)
    logger.info(f"{data[0]} imported successfully")

    logger.info(f"Run data quality tests for {data[0]}")
    run_data_quality_tests(data[1])
    logger.info(f"Data quality tests passed for {data[0]}")
    imported_data[data[0]] = data[1]
    logger.info("Import data complete")

    return imported_data

