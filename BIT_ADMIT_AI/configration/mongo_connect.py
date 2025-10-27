import sys
import os
import pymongo
import certifi
from BIT_ADMIT_AI.entity.config import SystemConfig
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.logger import logging

ca = certifi.where()


class MongoDbClient:
    """Creating a mongo db connection, uses the system config"""

    client = None

    def __init__(self, system_config: SystemConfig):
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
