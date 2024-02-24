import os
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch
from ....shared_package.shared.monitoring.model_metrics import log_metadata


@pytest.fixture
def model_metrics_path(tmpdir):
    return os.path.join(str(tmpdir), 'model_metrics.csv')


@pytest.fixture
def expected_metadata(metrics, metadata_cols):

    expected_metadata = {
        'project': 'example',
        'run_id': '123',
        'date_time_run': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': 'RandomForest'}
    metadata_dict_metrics = {'metric_name': list(metrics.keys()),
                             'metric_value': list(metrics.values())
                             }
    expected_metadata = dict(expected_metadata, **metadata_dict_metrics)
    return pd.DataFrame(data=expected_metadata, columns=metadata_cols)


@pytest.fixture
def old_metadata(metadata_cols):

    old_metadata = {
        'project': 'example',
        'run_id': '123',
        'date_time_run': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': 'RandomForest',
        'metric_name': 'accuracy',
        'metric_value': 34.0}

    return pd.DataFrame(old_metadata, columns=metadata_cols, index=[0])


@pytest.fixture
def metrics():
    return {'metric1': 0.9, 'metric2': 0.85}


@pytest.fixture
def metadata_cols():
    return ['project', 'run_id', 'date_time_run',
            'model_name', 'metric_name', 'metric_value']


@pytest.mark.parametrize("existing_metadata", [True, False])
@pytest.mark.parametrize("environment", ['local'])
@pytest.mark.unit
def test_log_metadata(model_metrics_path, expected_metadata, old_metadata, metrics,
                      metadata_cols, existing_metadata, environment):

    # Test when no previously logged metadata exists:
    with patch('os.path.normpath', return_value=model_metrics_path) as mock_normpath, \
         patch('os.path.isfile', return_value=False) as mock_isfile, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('pandas.read_csv', return_value=old_metadata if existing_metadata else
         pd.DataFrame(columns=metadata_cols)) as mock_read_csv:

        result = log_metadata(metrics, environment, model_metrics_path)

        assert list(result.columns) == list(expected_metadata.columns)
        assert len(result) == len(expected_metadata)

