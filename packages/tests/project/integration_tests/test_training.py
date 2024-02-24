# Install third-party packages:
from unittest.mock import patch
import pytest
import logging
import pandas as pd

# Install project packages:
from project.ml_ops.training import run_training


@pytest.fixture
def sample_logger():
    return logging.getLogger(__name__)


@pytest.fixture
def sample_config():
    return {
        'environment': 'local',
        'output_model_metrics_path': 'model_metrics/model_metrics.csv',
        'output_model_publish_name': 'model_rf'
    }


@pytest.fixture
def sample_training():
    data = {'Survived': [1, 0, 1, 0, 1],
            'Pclass': [1, 2, 3, 1, 2],
            'Age': [30, 25, 22, 35, 28],
            'Sex_male': [1, 0, 1, 0, 0],
            'Sex-female': [0, 1, 0, 1, 1]}
    return pd.DataFrame(data)


@pytest.fixture
def sample_metadata():
    data = {'project': ['example'],
            'run_id': ['0000001'],
            'date_time_run': ['2023-05-01'],
            'model_name': ['RandomForest'],
            'metric_name': ['accuracy'],
            'metric_value': [0.76]}
    return pd.DataFrame(data)


@pytest.mark.integration
def test_run_training_integration(sample_logger, sample_config, sample_training, caplog, sample_metadata):

    with caplog.at_level(logging.INFO):
        result = run_training(sample_logger, sample_config, sample_training)

    assert isinstance(result, pd.DataFrame)

    assert "Run training" in caplog.text
    assert "Split data" in caplog.text
    assert "Run model training" in caplog.text
    assert "Log model metadata" in caplog.text
    assert "Publish model to mlflow registry" in caplog.text
    assert "Training complete" in caplog.text

    sample_config['environment'] = 'non_existent_environment'
    with pytest.raises(ValueError, match=r"Environment .+ is not currently supported"):
        run_training(sample_logger, sample_config, sample_training)



