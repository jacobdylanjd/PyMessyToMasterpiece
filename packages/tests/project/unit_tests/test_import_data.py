# Install third-party packages:
from unittest.mock import patch
import pytest
import os

# Install project packages:
from ....project_package.project.ml_source.import_data import import_data
from ..data.test_data import sample_dataframe


@pytest.mark.unit
def test_import_data(sample_dataframe):

    # Test for environment equals local:
    config = {'environment': 'local',
              'input_file_path_titanic': '/folder/folder/titanic.csv'}

    with patch('pandas.read_csv', return_value=sample_dataframe) as mock_read_csv:
        name, df = import_data(config)

        mock_read_csv.assert_called_with(os.path.normpath(config['input_file_path_titanic']))
        assert name == 'titanic_data'
        assert list(df.columns.values) == list(sample_dataframe.columns.values)
        assert df.equals(sample_dataframe) is True

    config['input_file_path_titanic'] = '/folder/folder/titanic.csv'
    with pytest.raises(FileNotFoundError, match=r"File .+ not found. Please check the file path and try again."):
        import_data(config)

    # Test for environment which is not supported:
    config['environment'] = 'non_existent_environment'
    with pytest.raises(ValueError, match=r"Environment .+ is not currently supported"):
        import_data(config)

