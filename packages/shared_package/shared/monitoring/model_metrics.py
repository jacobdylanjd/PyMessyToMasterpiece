# Install third-party packages:
import os
import uuid
from datetime import datetime
import pandas as pd


def log_metadata(metrics: dict,
                 environment: str,
                 model_metrics_path: str) -> pd.DataFrame:
    """
    Log model training metadata to file.

    Args:
        metrics (dict): Training metrics to be logged.
        environment (str): Deployment environment.
        model_metrics_path (str): Path for model metrics.
    Return:
        metadata_temp (pd.DataFrame): model metadata for current run.
    """

    # Filter out model:
    metrics = {key: metrics[key] for key in metrics if key != 'model'}

    metadata_cols = ['project', 'run_id', 'date_time_run',
                     'model_name', 'metric_name', 'metric_value']

    if environment == 'local':
        # Check if file exists already exists if not create one:
        model_metrics_path = os.path.normpath(model_metrics_path)

        if not os.path.isfile(model_metrics_path):
            # Create the CSV file with specified columns
            df = pd.DataFrame(columns=metadata_cols)
            df.to_csv(model_metrics_path, index=False)
    else:
        raise ValueError(f"Environment {environment} is not currently supported")

    # Create temp metadata dataframe using metrics:
    metadata_dict_details = {
        'project': 'example',
        'run_id': str(uuid.uuid4()),
        'date_time_run': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': 'RandomForest'}

    metadata_dict_metrics = {'metric_name': list(metrics.keys()),
                             'metric_value': list(metrics.values())
                             }

    metadata_dict = dict(metadata_dict_details, **metadata_dict_metrics)

    # Create temp metadata dataframe:
    metadata_temp = pd.DataFrame(data=metadata_dict, columns=metadata_cols)

    if environment == 'local':
        # Update metadata csv with temp:
        metadata = pd.read_csv(model_metrics_path, usecols=metadata_cols)
        metadata = pd.concat([metadata, metadata_temp])
        metadata.to_csv(model_metrics_path, index=False)
    else:
        raise ValueError(f"Environment {environment} is not currently supported")

    return metadata_temp
