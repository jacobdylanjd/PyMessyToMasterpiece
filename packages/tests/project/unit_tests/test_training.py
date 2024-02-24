# Install third-party packages:
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Install project packages:
from ....project_package.project.ml_source.training import split_data, model_training
from ..data.test_data import sample_dataframe


@pytest.fixture
def sample_data():
    # Create sample data for testing
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}


@pytest.mark.unit
def test_model_training(sample_data):

    result = model_training(sample_data)

    assert isinstance(result, dict)

    assert 'model' in result
    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result

    assert isinstance(result['model'], RandomForestClassifier)

    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['precision'] <= 1
    assert 0 <= result['recall'] <= 1


def test_split_data(sample_dataframe):

    result = split_data(sample_dataframe)

    assert isinstance(result, dict)

    assert 'X_train' in result
    assert 'X_test' in result
    assert 'y_train' in result
    assert 'y_test' in result

    assert isinstance(result['X_train'], pd.DataFrame)
    assert isinstance(result['X_test'], pd.DataFrame)
    assert isinstance(result['y_train'], pd.Series)
    assert isinstance(result['y_test'], pd.Series)

    assert result['X_train'].shape[0] + result['X_test'].shape[0] == sample_dataframe.shape[0]
    assert result['y_train'].shape[0] + result['y_test'].shape[0] == sample_dataframe.shape[0]