import json
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="evidently")


import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame

from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.utils.main_utils import read_yaml_file, write_yaml_file
from BIT_ADMIT_AI.entity.artifact import DataValidationArtifact, DAArtifacts
from BIT_ADMIT_AI.entity.config import DataValidationConfig
from BIT_ADMIT_AI.constant import SCHEMA_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DAArtifacts,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_PATH)
        except Exception as e:
            logging.error(f"failing to start the data validation, {e}")
            raise BitAdmitAIException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Failed to read data from {file_path}: {e}")
            raise BitAdmitAIException(e, sys) from e

    def validate_num_of_col(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            logging.error(f"Failing to validate the num of columns, {e}")
            raise BitAdmitAIException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_num_columns = []
            missing_cat_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_num_columns.append(column)

            if len(missing_num_columns) > 0:
                logging.info(f"Missing numerical column: {missing_num_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_cat_columns.append(column)

            if len(missing_cat_columns) > 0:
                logging.info(f"Missing categorical column: {missing_cat_columns}")

            return (
                False
                if len(missing_cat_columns) > 0 or len(missing_num_columns) > 0
                else True
            )
        except Exception as e:
            logging.error(f"failed to check the col existing {e}")
            raise BitAdmitAIException(e, sys) from e

    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
    ) -> bool:
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(reference_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report,
            )

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"][
                "n_drifted_features"
            ]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise BitAdmitAIException(e, sys) from e

    def init_data_validation(self) -> DataValidationArtifact:

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.training_file_path
                ),
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                ),
            )

            status = self.validate_num_of_col(dataframe=train_df)
            logging.info(
                f"All required columns present in training dataframe: {status}"
            )
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_num_of_col(dataframe=test_df)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_column_exist(df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise BitAdmitAIException(e, sys) from e
