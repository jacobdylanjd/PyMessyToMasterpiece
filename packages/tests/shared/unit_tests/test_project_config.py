# Install third-party packages:
from unittest.mock import patch, mock_open
import pytest

# Install project packages:
from ....shared_package.shared.config.project_config import load_config


@pytest.mark.parametrize("environment, input_file_path", [('local', '/folder/folder/titanic.csv')])
def test_load_config(environment, input_file_path):
    yaml_text = f"""
        local:
            input_file_path_titanic: {input_file_path}
        """

    with patch('os.environ.get', return_value=environment) as mock_environment:
        with patch("builtins.open", mock_open(read_data=yaml_text)) as mock_file:
            config = load_config('test_project')

            mock_environment.assert_called_with('ENVIRONMENT')
            assert config['environment'] == environment
            assert config['project_name'] == 'test_project'
            assert config['input_file_path_titanic'] == input_file_path


@pytest.mark.unit
def test_load_config_environment_not_supported():
    with patch('os.environ.get', return_value='non_existent_environment') as mock_environment:
        with pytest.raises(ValueError,
                           match=rf"Environment .+ not accounted for in load_config, please update function."):
            load_config('test_project')

