import pandas as pd
import os


def import_data() -> pd.DataFrame:
    """Import data for project.

        Args: None
        Return:
            df (pd.DataFrame): Imported data.
    """

    # Import csv from local path:
    file_name = 'titanic.csv'
    path = os.path.join('data/', file_name)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found. Please check the file path and try again.") from None

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
