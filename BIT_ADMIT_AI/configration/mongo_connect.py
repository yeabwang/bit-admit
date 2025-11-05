"""MongoDB connection utility.

Provides MongoDbClient, a thin wrapper around a shared pymongo.MongoClient.
- Reads connection info from SystemConfig (MONGODB_URL_KEY, DATABASE_NAME).
- Uses certifi CA bundle for TLS.
- Raises BitAdmitAIException on misconfiguration or connection failures.
"""

import sys
import os
import pymongo
import certifi
from BIT_ADMIT_AI.entity.config import SystemConfig
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging

ca = certifi.where()


class MongoDbClient:
    """Create and manage a shared MongoDB client connection.

    Attributes:
        client: shared pymongo.MongoClient instance.
        database:  Database object for SystemConfig.DATABASE_NAME.
        database_name: Name of the active database.
    """

    client = None

    def __init__(self, system_config: SystemConfig):
        """Initialize the MongoDB client from SystemConfig.

        Creates a singleton pymongo.MongoClient and
        binds the configured database.

        Args:
            system_config: Provides MONGODB_URL_KEY and DATABASE_NAME.

        Raises:
            BitAdmitAIException: If MONGODB_URL_KEY is missing/empty or connection fails.
        """
        try:
            if not MongoDbClient.client:
                self.mongo_db_url = system_config.MONGODB_URL_KEY
                if not self.mongo_db_url:
                    logging.error("Mongodb url key is not set check the env and set")
                    raise BitAdmitAIException(
                        "'MONGODB_URL_KEY' is not set correctly in the .env", sys
                    )
                else:
                    MongoDbClient.client = pymongo.MongoClient(
                        self.mongo_db_url, tlsCAFile=ca
                    )

            self.client = MongoDbClient.client
            self.database = self.client[system_config.DATABASE_NAME]
            self.database_name = system_config.DATABASE_NAME

            logging.info("DB connection made successfully.")

        except Exception as e:
            logging.error(f"Error occurred during connection: {e}")
            raise BitAdmitAIException(e, sys)
