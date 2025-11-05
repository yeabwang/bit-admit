"""Configuration dataclasses for the pipeline.

Centralizes paths, filenames through:
ingestion, validation, transformation, training, evaluation, and model push.

Notes:
- TIME_STAMP is embedded in artifact_dir to keep runs isolated.
- Environment-driven constants are imported from BIT_ADMIT_AI.constant.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from BIT_ADMIT_AI.constant import *

TIME_STAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class PTrainingConfig:
    """Training pipeline configuration.

    Attributes:
        p_name: Pipeline name.
        artifact_dir: Root directory for all artifacts of this run.
        current_time: Timestamp used to version artifacts.
    """

    p_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIME_STAMP)
    current_time: str = TIME_STAMP


training_config: PTrainingConfig = PTrainingConfig()


@dataclass
class DataIngestionConfig:
    """Data ingestion configuration.

    Attributes:
        ingestion_dir: Base dir for ingestion artifacts.
        feature_store_dir: Path to the raw feature store CSV.
        training_file_path: Path to the train split CSV.
        test_file_path: Path to the test split CSV.
        test_ratio: Fraction for test split.
        collection_name: MongoDB collection name to read from.
    """

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
    """System/environment configuration.

    Attributes:
        DATABASE_NAME: Default MongoDB database name.
        COLLECTION_NAME: Default MongoDB collection name.
        MONGODB_URL_KEY: Env var key containing the Mongo connection string.
    """

    DATABASE_NAME = DATABASE_NAME
    COLLECTION_NAME = COLLECTION_NAME
    MONGODB_URL_KEY = MONGODB_URL_KEY


@dataclass
class DataValidationConfig:
    """Data validation configuration.

    Attributes:
        data_validation_dir: Base dir for validation artifacts.
        drift_report_file_path: Path to the data drift report file.
    """

    data_validation_dir: str = os.path.join(
        training_config.artifact_dir, DATA_VAL_DIR_NAME
    )
    drift_report_file_path: str = os.path.join(
        data_validation_dir, DATA_DRIFT_REPORT_DIR, DATA_DRIFT_REPORT_FILE_NAME
    )


@dataclass
class DataTransformationConfig:
    """Data transformation configuration.

    Attributes:
        data_transformation_dir: Base dir for transformation artifacts.
        transformed_train_file_path: Path to transformed train (.npy).
        transformed_test_file_path: Path to transformed test (.npy).
        transformed_object_file_path: Path to the fitted preprocessing object.
    """

    data_transformation_dir: str = os.path.join(
        training_config.artifact_dir, DATA_TRANS_DIR_NAME
    )
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANS_TRANSFORMED_DATA_DIR,
        TRAIN_FILE_NAME.replace("csv", "npy"),
    )
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANS_TRANSFORMED_DATA_DIR,
        TEST_FILE_NAME.replace("csv", "npy"),
    )
    transformed_object_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANS_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJ_FILE,
    )


@dataclass
class ModelTrainerConfig:
    """Model trainer configuration.

    Attributes:
        model_trainer_dir: Base dir for training artifacts.
        trained_model_file_path: Path to persist the trained model.
        expected_accuracy: Minimum acceptable score to consider the model valid.
        model_config_file_path: Path to model hyperparameter config.
    """

    model_trainer_dir: str = os.path.join(
        training_config.artifact_dir, MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


@dataclass
class ModelEvaluationConfig:
    """Model evaluation configuration.

    Attributes:
        best_model_dir: Directory containing the production/best model.
        best_model_path: Path to the best model file.
        best_model_metrics_path: Path to metrics of the best model.
        change_threshold: Minimum improvement required to accept a new model.
    """

    best_model_dir: str = BEST_MODEL_DIR
    best_model_path: str = BEST_MODEL_PATH
    best_model_metrics_path: str = BEST_MODEL_METADATA_PATH
    change_threshold: float = EVAL_CHANGED_THRESHOLD_SCORE


@dataclass
class ModelPusherConfig:
    """Model pusher configuration.

    Attributes:
        best_model_dir: Directory to write/promote the best model.
        best_model_path: Path where the promoted model will be saved.
        best_model_metrics_path: Path to the promoted model's metrics file.
    """

    best_model_dir: str = BEST_MODEL_DIR
    best_model_path: str = BEST_MODEL_PATH
    best_model_metrics_path: str = BEST_MODEL_METADATA_PATH
