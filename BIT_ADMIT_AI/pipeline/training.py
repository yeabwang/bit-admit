import sys
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.components.data_ingestion import DataIngestion
from BIT_ADMIT_AI.components.data_validation import DataValidation
from BIT_ADMIT_AI.entity.config import DataIngestionConfig, DataValidationConfig
from BIT_ADMIT_AI.entity.artifact import DAArtifacts, DataValidationArtifact


class TrainingPipline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_val_config = DataValidationConfig()

    def start_ingestion(self):
        try:
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            ingested_artifact = data_ingestion.init_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            return ingested_artifact
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DAArtifacts
    ) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_val_config,
            )

            data_validation_artifact = data_validation.init_data_validation()

            logging.info("Performed the data validation operation")

            return data_validation_artifact

        except Exception as e:
            raise BitAdmitAIException(e, sys) from e

    def run_pipeline(
        self,
    ) -> None:
        try:
            data_ingestion_artifact = self.start_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )
        except Exception as e:
            raise BitAdmitAIException(e, sys)
