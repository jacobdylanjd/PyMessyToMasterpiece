import yaml
import os


def load_config(project_name: str) -> dict:

    environment = os.environ.get('ENVIRONMENT')

    # *********************
    # Note in real world solution, we should store config.yml in external storage location, and download the file
    # to local storage to load, this ensures a single source of truth for config.yml
    # In this case, we would not need to use if statements below, as all environments would handle the config file the
    # same
    # *********************

    if environment == 'local':
        config_path = 'config.yml'
        try:
            with open(config_path) as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File at path {config_path} not found. Please create config.yml file.")

        config = config[environment]

    # Add elif statements to handle additional environments..

    else:
        raise ValueError(f"Environment {environment} not accounted for in load_config, please update function.")

    # Set the project name and environment in the config file:
    config['project_name'] = project_name
    config['environment'] = environment

    return config
