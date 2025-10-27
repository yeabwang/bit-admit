import sys
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.components.data_ingestion import DataIngestion
from BIT_ADMIT_AI.entity.config import DataIngestionConfig
from BIT_ADMIT_AI.entity.artifact import DAArtifacts


class TrainingPipline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_ingestion(self):
        try:
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            ingested_artifact = data_ingestion.init_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            return ingested_artifact
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def run_pipeline(
        self,
    ) -> None:
        try:
            data_ingestion_artifact = self.start_ingestion()
        except Exception as e:
            raise BitAdmitAIException(e, sys)
