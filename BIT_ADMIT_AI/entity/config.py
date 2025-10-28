import os
from dataclasses import dataclass
from datetime import datetime
from BIT_ADMIT_AI.constant import *

TIME_STAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class PTrainingConfig:
    "Training pipline configration"

    p_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIME_STAMP)
    current_time: str = TIME_STAMP


training_config: PTrainingConfig = PTrainingConfig()


@dataclass
class DataIngestionConfig:
    "Config of ingestion"

    ingestion_dir: str = os.path.join(training_config.artifact_dir, DA_DIR_NAME)
    feature_store_dir: str = os.path.join(
        ingestion_dir, DA_FEATURE_STORE_DIR, FILE_NAME
    )
    training_file_path: str = os.path.join(
        ingestion_dir, DA_INGESTED_DIR, TRAIN_FILE_NAME
    )
    test_file_path: str = os.path.join(ingestion_dir, DA_INGESTED_DIR, TEST_FILE_NAME)
    test_ratio: float = DA_TRAIN_TEST_TEST_RATIO
    collection_name: str = DA_COLLECTION_NAME


@dataclass
class SystemConfig:
    DATABASE_NAME = DATABASE_NAME
    COLLECTION_NAME = COLLECTION_NAME
    MONGODB_URL_KEY = MONGODB_URL_KEY


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_config.artifact_dir, DATA_VAL_DIR_NAME
    )
    drift_report_file_path: str = os.path.join(
        data_validation_dir, DATA_DRIFT_REPORT_DIR, DATA_DRIFT_REPORT_FILE_NAME
    )
