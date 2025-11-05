"""MongoDB data access utilities.

Provides DataAccessAndHandling to pull MongoDB collections into pandas DataFrames.
- Connects via MongoDbClient using SystemConfig.
- Drops MongoDB "_id" and normalizes "na" to NaN.
- Raises BitAdmitAIException on failures.
"""

import pandas as pd
import numpy as np
import sys
from typing import Optional

from BIT_ADMIT_AI.configration.mongo_connect import MongoDbClient
from BIT_ADMIT_AI.entity.config import SystemConfig
from BIT_ADMIT_AI.exceptions import BitAdmitAIException


class DataAccessAndHandling:
    """Thin wrapper over MongoDbClient to read collections as DataFrames.

    Args:
        system_config: Configuration providing DATABASE_NAME and connection params.

    Attributes:
        database_name: Default database name used when not overridden.
        mongo_client: Connected MongoDbClient instance.
    """

    def __init__(self, system_config: SystemConfig = SystemConfig()):
        self.database_name = system_config.DATABASE_NAME

        try:
            self.mongo_client = MongoDbClient(SystemConfig())
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def collection_to_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a MongoDB collection into a pandas DataFrame.

        Drops "_id" if present and replaces string "na" with NaN.

        Args:
            collection_name: Name of the collection to read.
            database_name: Optional database override; defaults to the configured database.

        Returns:
            pandas.DataFrame: Collection data.

        Raises:
            BitAdmitAIException: On connection or read errors.
        """
        try:
            target_db = database_name or self.database_name
            collection = self.mongo_client.client[target_db][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise BitAdmitAIException(e, sys)
