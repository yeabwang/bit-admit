"""Centralized constants for BIT_ADMIT_AI.

Loads environment variables from a .env at repo root and exposes
readable names for pipeline components (ingestion, validation,
transformation, training, evaluation, model push, FastAPI).

Environment:
- DATABASE_NAME
- COLLECTION_NAME
- MONGODB_URL_KEY

Note:
- find_dotenv(raise_error_if_not_found=True) will raise if .env is missing.
"""

import os
from dotenv import find_dotenv, load_dotenv
from typing import List

# Load .env if present; don't fail hard if it's missing so the pipeline can fall back to local CSVs.
_dotenv_path = find_dotenv(raise_error_if_not_found=False)
if _dotenv_path:
	load_dotenv(_dotenv_path)

# Overall system constants
DATABASE_NAME: str = os.getenv("DATABASE_NAME", "")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "")
MONGODB_URL_KEY: str = os.getenv("MONGODB_URL_KEY", "")
PIPELINE_NAME: str = "bit_admit_ai"
ARTIFACT_DIR: str = "bit_artifact"
MODEL_FILE_NAME: str = "model.pkl"
FILE_NAME = "admission_data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TARGET_COLUMNS: List[str] = ["admission_decision", "scholarship_tier"]
PREPROCESSING_OBJ_FILE: str = "preprocessing.pkl"
SCHEMA_PATH = os.path.join("config", "schema.yaml")

# MongoDB ingestion
DA_COLLECTION_NAME: str = COLLECTION_NAME
DA_DIR_NAME: str = os.getenv("COLLECTION_NAME", "unknown_collection").lower()
DA_FEATURE_STORE_DIR: str = "feature_store"
DA_INGESTED_DIR: str = "ingested_data"
DA_TRAIN_TEST_TEST_RATIO: float = 0.2


# Data validation
DATA_VAL_DIR_NAME: str = "data_validation"
DATA_DRIFT_REPORT_DIR: str = "drift_report"
DATA_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# Data transformation
DATA_TRANS_DIR_NAME: str = "data_transformation"
DATA_TRANS_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANS_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model training
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.9
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# Model evaluation # The value is chosen with gut feeling, feel free to change it depending on use
EVAL_CHANGED_THRESHOLD_SCORE: float = 0.07


# Model pusher (currently saved locally but will be the place to look during deployment)
BEST_MODEL_DIR: str = "best_model"
BEST_MODEL_FILE: str = "model.pkl"
BEST_MODEL_METADATA_FILE: str = "metrics.yaml"
BEST_MODEL_PATH: str = os.path.join(BEST_MODEL_DIR, BEST_MODEL_FILE)
BEST_MODEL_METADATA_PATH: str = os.path.join(BEST_MODEL_DIR, BEST_MODEL_METADATA_FILE)

# FastAPI
APP_HOST = "0.0.0.0"
APP_PORT = 8080
