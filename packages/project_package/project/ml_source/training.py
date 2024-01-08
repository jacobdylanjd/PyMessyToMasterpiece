# Install third-party packages:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import uuid
from datetime import datetime
import pickle
import mlflow


def model_training(data: dict) -> dict:
    """Run supervised learning model training and testing.

        Args:
            data (dict): Data split for model train/test.
        Return:
            out (dict): Trained model and associated metadata.
    """

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Train the Random Forest Classifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the testing set:
    y_pred = model.predict(X_test)

    # Calculate the model performance metrics:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    out = {'model': model,
           'accuracy': accuracy,
           'precision': precision,
           'recall': recall}

    return out


def split_data(df: pd.DataFrame) -> dict:
    """Split data into train and test datasets.

        Args:
            df (pd.DataFrame): Data to be split for model train/test.
        Return:
            data (dict): Data split for model train/test.
    """

    # Create the feature df and target variable:
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Split the dataset into training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    data = {'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test}

    return data


def log_metadata(config: dict, metrics: dict) -> pd.DataFrame:
    """Log model training metadata to file.

        Args:
            config (dict): Project configuration file.
            metrics (dict): Training metrics to be logged.
        Return:
            metadata_temp (pd.DataFrame): model metadata for current run.
    """

    # Filter out model:
    metrics = {key: metrics[key] for key in metrics if key != 'model'}

    metadata_cols = ['project', 'run_id', 'date_time_run',
                     'model_name', 'metric_name', 'metric_value']

    if config['environment'] == 'local':
        # Check if file exists already exists if not create one:
        model_metrics_path = os.path.normpath(config['output_model_metrics_path'])

        if not os.path.isfile(model_metrics_path):
            # Create the CSV file with specified columns
            df = pd.DataFrame(columns=metadata_cols)
            df.to_csv(model_metrics_path, index=False)
    else:
        raise ValueError(f"Environment {config['environment']} is not currently supported")

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

    if config['environment'] == 'local':
        # Update metadata csv with temp:
        metadata = pd.read_csv(model_metrics_path, usecols=metadata_cols)
        metadata = pd.concat([metadata, metadata_temp])
        metadata.to_csv(model_metrics_path, index=False)
    else:
        raise ValueError(f"Environment {config['environment']} is not currently supported")

    return metadata_temp

