# Install third-party packages:
import pytest
import logging
import pandas as pd

# Install project packages:
from project.ml_ops.feature_engineering import run_feature_engineering
from ..data.test_data import sample_dataframe


@pytest.fixture
def sample_logger():
    return logging.getLogger(__name__)


@pytest.fixture
def sample_config():
    return {'environment': 'local', 'input_file_path_titanic': '/folder/folder/titanic.csv'}


@pytest.mark.integration
def test_run_feature_engineering(sample_logger, sample_dataframe, caplog):

    with caplog.at_level(logging.INFO):
        data = {'titanic_data': sample_dataframe}
        result_df = run_feature_engineering(sample_logger, data)

        assert isinstance(result_df, pd.DataFrame)
        assert set(result_df.columns) == set(['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                                              'Sex_female', 'Sex_male'])

        assert "Run feature engineering" in caplog.text
        assert "Drop columns" in caplog.text
        assert "Drop NA" in caplog.text
        assert "Run one-hot-encoding" in caplog.text
        assert "Feature engineering complete" in caplog.text

