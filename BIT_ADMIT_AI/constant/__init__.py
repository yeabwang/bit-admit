import os
from dotenv import find_dotenv, load_dotenv
from typing import List

find_dotenv(raise_error_if_not_found=True)
load_dotenv()

"Overall system constants"
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

"Related to mongo ingestion"
DA_COLLECTION_NAME: str = COLLECTION_NAME
DA_DIR_NAME: str = os.getenv("COLLECTION_NAME", "unknown_collection").lower()
DA_FEATURE_STORE_DIR: str = "feature_store"
DA_INGESTED_DIR: str = "ingested_data"
DA_TRAIN_TEST_TEST_RATIO: float = 0.2


"Related to validation"
DATA_VAL_DIR_NAME: str = "data_validation"
DATA_DRIFT_REPORT_DIR: str = "drift_report"
DATA_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"Related to data transformation"
DATA_TRANS_DIR_NAME: str = "data_transformation"
DATA_TRANS_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANS_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"Related to model training"
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.9
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

"Related to model evaluation"
EVAL_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = ""
MODEL_PUSHER_S3_KEY = ""
