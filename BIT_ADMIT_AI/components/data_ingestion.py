import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from BIT_ADMIT_AI.entity.config import DataIngestionConfig
from BIT_ADMIT_AI.entity.artifact import DAArtifacts
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.data_access.data_access import DataAccessAndHandling


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def export_to_feature_store(self):
        try:
            logging.info(f"Exporting data from mongodb")
            admission_data = DataAccessAndHandling()
            dataframe = admission_data.collection_to_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_dir
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data check here: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def dataset_split(self, dataframe: DataFrame) -> None:
        logging.info("Entered dataset_split method")
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.test_ratio
            )
            logging.info("Performed train test split")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.test_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise BitAdmitAIException(e, sys) from e

    def init_data_ingestion(self) -> DAArtifacts:
        try:
            dataframe = self.export_to_feature_store()

            logging.info("Got the data from mongodb")

            self.dataset_split(dataframe)

            logging.info("Performed train test split on the dataset")

            data_ingestion_artifact = DAArtifacts(
                training_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise BitAdmitAIException(e, sys) from e
