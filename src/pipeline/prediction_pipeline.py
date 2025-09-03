from src.utils import load_object
from src.components.model_trainer import model_trainer_config
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging

model=load_object(model_trainer_config.trained_model_path)

class Prediction:
    def __init__(self,input_data):
        self.prediction=None
        self.input_data=input_data
    
    def predict(self):
        try:
            self.prediction=model.predict(pd.DataFrame([self.input_data]))
            logging.info("prediction successfull")
            return self.prediction
        except Exception as e:
            raise CustomException(e,sys)