# Install third-party packages:
import logging


def setup_logging(module_name: str) -> logging.Logger:
    """
    Setup python logging so that logs print to console.

    Args:
        module_name (str): The module name to use in logger.
    Return:
        None
    """

    # Create a logger object:
    logger = logging.getLogger(module_name)

    # Set the logging level:
    logger.setLevel(logging.DEBUG)

    # Create a StreamHandler object:
    console_handler = logging.StreamHandler()

    # Set the logging level for the console handler:
    console_handler.setLevel(logging.DEBUG)

    # Create a Formatter object:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Add the formatter to the console handler:
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger:
    logger.addHandler(console_handler)

    logger.info("Logger setup")

    return logger

