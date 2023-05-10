from modules.ml_ops.setup_logging import setup_logging
from modules.ml_ops.import_data import run_import_data
from modules.ml_ops.feature_engineering import run_feature_engineering
from modules.ml_ops.training import run_training


# Run python logging setup:
logger = setup_logging(__name__)

# Run data import:
try:
    df = run_import_data(logger)
except Exception as e:
    logger.exception(f"In import data - {e}")
    raise

# Run feature engineering:
try:
    df = run_feature_engineering(logger, df)
except Exception as e:
    logger.exception(f"In feature engineering - {e}")
    raise

# Run model training:
try:
    metadata = run_training(logger, df)
except Exception as e:
    logger.exception(f"In training - {e}")
    raise

print(metadata)
