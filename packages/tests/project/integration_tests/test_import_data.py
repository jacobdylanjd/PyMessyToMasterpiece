# Install third-party packages:
from unittest.mock import patch
import pytest
import logging

# Install project packages:
from project.ml_ops.import_data import run_import_data
from ..data.test_data import sample_dataframe


@pytest.fixture
def sample_logger():
    return logging.getLogger(__name__)


@pytest.fixture
def sample_config():
    return {'environment': 'local', 'input_file_path_titanic': '/folder/folder/titanic.csv'}


@pytest.mark.integration
def test_run_import_data(sample_logger, sample_config, caplog):

    with caplog.at_level(logging.INFO):
        with patch('project.ml_ops.import_data.import_data', return_value=('titanic_data', sample_dataframe)) \
                as mock_import_data:
            with patch('project.ml_ops.import_data.run_data_quality_tests', return_value=None) \
                    as run_data_quality_tests:
                result = run_import_data(sample_logger, sample_config)

    mock_import_data.assert_called_once()
    run_data_quality_tests.assert_called_once()

    assert isinstance(result, dict)
    assert 'titanic_data' in result

    assert "Run import data" in caplog.text
    assert "titanic_data imported successfully" in caplog.text
    assert "Run data quality tests for titanic_data" in caplog.text
    assert "Data quality tests passed for titanic_data" in caplog.text
    assert "Import data complete" in caplog.text

    sample_config['environment'] = 'non_existent_environment'
    with pytest.raises(ValueError, match=r"Environment .+ is not currently supported"):
        run_import_data(sample_logger, sample_config)

