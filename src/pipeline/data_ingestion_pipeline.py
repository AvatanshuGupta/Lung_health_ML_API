from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import sys

if __name__ == "__main__":
    try:
        logging.info("Starting data ingestion pipeline...")
        di=DataIngestion()
        di.initiate_data_ingestion()
        logging.info("data ingestion pipeline excecuted")
    except Exception as e:
        logging.error("Error in data ingestion pipeline.")
        raise CustomException(e,sys)
