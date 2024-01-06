import pandas as pd
import os


def import_data(config: dict) -> pd.DataFrame:
    """Import data for project.

        Args: config (dict): Project configuration file.
        Return:
            df (pd.DataFrame): Imported data.
    """

    if config['environment'] == 'local':
        try:
            path = os.path.normpath(config['input_file_path_titanic'])
            file_name = os.path.basename(path)
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_name} not found. Please check the file path and try again.") from None
    else:
        raise ValueError(f"Environment {config['environment']} is not currently supported")

    return df


def run_data_quality_tests(df: pd.DataFrame) -> None:
    """Run data quality tests.

        Args:
            df (pd.DataFrame): Data to quality test.
        Return:
            None
    """

    # Run data quality checks:
    assert len(df.drop_duplicates()) == len(df), "Duplicates present in data"

    return None
