# Install project packages:
from project.ml_ops.setup_logging import setup_logging
from project.ml_ops.import_data import run_import_data
from project.ml_ops.feature_engineering import run_feature_engineering
from project.ml_ops.training import run_training
from project.ml_ops.config import load_config

# Install third-party packages:
import mlflow

# Run python logging setup:
logger = setup_logging(__name__)

# Load project config file:
project_name = "demo_project"
config = load_config(project_name)

# Set up mlflow experiment:
mlflow.set_experiment(config['experiment_path'])
experiment = mlflow.get_experiment_by_name(config['experiment_path'])
experiment_id = experiment.experiment_id
logger.info(f"Start mlflow run, experiment_id: {experiment_id}")
mlflow.start_run(experiment_id=experiment_id, run_name=config['project_name'])

# Run data import:
try:
    df = run_import_data(logger, config)
except Exception as e:
    logger.exception(f"In import data - {e}")
    logger.info(f"End mlflow run, experiment_id: {experiment_id}")
    mlflow.end_run()
    raise

# Run feature engineering:
try:
    df = run_feature_engineering(logger, df)
except Exception as e:
    logger.exception(f"In feature engineering - {e}")
    mlflow.end_run()
    raise

# Run model training:
try:
    metadata = run_training(logger, config, df)
except Exception as e:
    logger.exception(f"In training - {e}")
    mlflow.end_run()
    raise

print(metadata)

logger.info(f"Orchestrator script completed successfully")
logger.info(f"End mlflow run, experiment_id: {experiment_id}")
mlflow.end_run()
