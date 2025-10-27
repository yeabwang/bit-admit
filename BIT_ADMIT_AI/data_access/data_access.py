import pandas as pd
import numpy as np
import sys
from typing import Optional

from BIT_ADMIT_AI.configration.mongo_connect import MongoDbClient
from BIT_ADMIT_AI.entity.config import SystemConfig
from BIT_ADMIT_AI.exceptions import BitAdmitAIException


class DataAccessAndHandling:
    def __init__(self, system_config: SystemConfig = SystemConfig()):
        self.database_name = system_config.DATABASE_NAME

        try:
            self.mongo_client = MongoDbClient(SystemConfig())
        except Exception as e:
            raise BitAdmitAIException(e, sys)

    def collection_to_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """For flexability allowing to pass db name with fallback to the default"""
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
