"""Data ingestion component.

Pulls admissions data from MongoDB into a feature-store CSV and creates train/test splits.

- export_to_feature_store: read MongoDB and persist raw CSV.
- dataset_split: split DataFrame and persist train/test CSVs.
- init_data_ingestion: orchestrate ingestion and return DAArtifacts.
- Writes CSVs to paths defined in DataIngestionConfig.
- Logs progress.

Raises:
- BitAdmitAIException: On any IO/DB/splitting error.
"""

import os
import sys
import glob
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from BIT_ADMIT_AI.entity.config import DataIngestionConfig, SystemConfig
from BIT_ADMIT_AI.entity.artifact import DAArtifacts
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.data_access.data_access import DataAccessAndHandling


class DataIngestion:
    """Ingest data from MongoDB and persist local artifacts.

    Args:
        data_ingestion_config: Paths and parameters for ingestion outputs.

    Attributes:
        data_ingestion_config: Stored ingestion configuration.
    """

    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """Initialize the ingestion component.

        Args:
            data_ingestion_config: Paths and parameters for ingestion outputs.

        Raises:
            BitAdmitAIException: If configuration initialization fails.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def export_to_feature_store(self):
        """Export data to the feature store CSV with MongoDB→local fallback.

        Priority:
        1) If MONGODB_URL_KEY is provided, try reading from MongoDB.
        2) If not provided or MongoDB access fails, fall back to local CSV in
           the repository's 'original_dataset' directory (pick latest *.csv).

        Returns:
            pandas.DataFrame: Raw admissions data used for downstream steps.

        Raises:
            BitAdmitAIException: If neither MongoDB nor local CSV can be read.
        """
        try:
            dataframe: DataFrame

            sys_cfg = SystemConfig()
            mongo_url = sys_cfg.MONGODB_URL_KEY

            if mongo_url:
                logging.info("Attempting to export data from MongoDB")
                try:
                    admission_data = DataAccessAndHandling()
                    dataframe = admission_data.collection_to_dataframe(
                        collection_name=self.data_ingestion_config.collection_name
                    )
                    logging.info(
                        f"MongoDB export successful. DataFrame shape: {dataframe.shape}"
                    )
                except Exception as e:
                    logging.warning(
                        f"MongoDB ingestion failed ({e}). Falling back to local CSV from 'original_dataset'."
                    )
                    dataframe = self._load_local_dataset()
            else:
                logging.info(
                    "MONGODB_URL_KEY is not set. Using local CSV from 'original_dataset' as fallback."
                )
                dataframe = self._load_local_dataset()

            # Persist to feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_dir
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data to feature store: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def _load_local_dataset(self) -> DataFrame:
        """Load the most recent CSV from the original_dataset folder.

        Returns:
            pandas.DataFrame: Data loaded from the local CSV.

        Raises:
            BitAdmitAIException: If the folder or a CSV file cannot be found/read.
        """
        try:
            # Resolve repo root relative to this file: BIT_ADMIT_AI/components/ -> repo root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            dataset_dir = os.path.join(repo_root, "original_dataset")

            if not os.path.isdir(dataset_dir):
                raise BitAdmitAIException(
                    f"Local dataset directory not found: {dataset_dir}", sys
                )

            csv_pattern = os.path.join(dataset_dir, "*.csv")
            csv_files = glob.glob(csv_pattern)
            if not csv_files:
                raise BitAdmitAIException(
                    f"No CSV files found under: {dataset_dir}. Please add a CSV to proceed.",
                    sys,
                )

            # Pick the most recently modified CSV
            latest_csv = max(csv_files, key=os.path.getmtime)
            logging.info(f"Loading local dataset from: {latest_csv}")
            df = pd.read_csv(latest_csv)
            logging.info(f"Local CSV load successful. DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def dataset_split(self, dataframe: DataFrame) -> None:
        """Split the dataset into train and test and persist CSVs.

        Args:
            dataframe: Input DataFrame to split.

        Side Effects:
            Writes train/test CSVs to training_file_path and test_file_path.

        Raises:
            BitAdmitAIException: If split or file write fails.
        """
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

            logging.info("Exported train and test file path.")
        except Exception as e:
            raise BitAdmitAIException(e, sys) from e

    def init_data_ingestion(self) -> DAArtifacts:
        """Run the ingestion workflow: export and split.

        Returns:
            DAArtifacts: Paths to persisted train and test files.

        Raises:
            BitAdmitAIException: On any step failure.
        """
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
