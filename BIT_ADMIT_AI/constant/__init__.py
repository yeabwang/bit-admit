import os
from dotenv import find_dotenv, load_dotenv

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

"Related to mongo ingestion"
DA_COLLECTION_NAME: str = ""
DA_DIR_NAME: str = os.getenv("COLLECTION_NAME", "unknown_collection").lower()
DA_FEATURE_STORE_DIR: str = "feature_store"
DA_INGESTED_DIR: str = "ingested_data"
DA_TRAIN_TEST_TEST_RATIO: float = 0.2
