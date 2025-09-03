from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sqlalchemy import create_engine, inspect
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import sys
from sklearn.model_selection import train_test_split

load_dotenv()
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the data ingestion method")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            DATABASE_URL=os.environ.get('DATABASE_URL')
            engine = create_engine(DATABASE_URL)
            df_diabetes = pd.read_sql("SELECT * FROM lung;", engine)
            logging.info("Data downloaded from database")
            train_data,test_data=train_test_split(df_diabetes,test_size=0.2,random_state=42)
            logging.info("train_test_split completed")
            train_data.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            df_diabetes.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            logging.info("Data ingestion process completed")
            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e,sys)
    


