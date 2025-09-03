from fastapi import FastAPI
from src.DataSchema.schema import Patient
from src.pipeline.prediction_pipeline import Prediction
from src.logger import logging
import sys
from src.exception import CustomException
app=FastAPI()
@app.get('/')
def welcome():
    return {'message':'Welocome to lung health prediction model api.'}

@app.post('/predict')
def predict(patient:Patient):
    input_data = {
    'GENDER': patient.GENDER,
    'AGE': patient.AGE,
    'SMOKING': patient.SMOKING,
    'YELLOW_FINGERS': patient.YELLOW_FINGERS,
    'ANXIETY': patient.ANXIETY,
    'FATIGUE': patient.FATIGUE,
    'ALLERGY': patient.ALLERGY,
    'WHEEZING': patient.WHEEZING.value, 
    'ALCOHOL CONSUMING': patient.ALCOHOL_CONSUMING,
    'COUGHING': patient.COUGHING,
    'SHORTNESS OF BREATH': patient.SHORTNESS_OF_BREATH,
    'SWALLOWING DIFFICULTY': patient.SWALLOWING_DIFFICULTY,
    'CHEST PAIN': patient.CHEST_PAIN
}

    try:
        pred_obj=Prediction(input_data)
        pred=pred_obj.predict()
        logging.info("prediction done successfully in api")
        return {"prediction":int(pred[0])}
    except Exception as e:
        raise CustomException(e,sys)
