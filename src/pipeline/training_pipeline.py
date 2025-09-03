from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.utils import load_csv
from src.logger import logging

# creating Data Ingestion obj
DIobj=DataIngestion()
logging.info("data ingestion obj created")

# getting train test data
train_data_path=DIobj.ingestion_config.train_data_path
test_data_path=DIobj.ingestion_config.test_data_path

# loading data
train_data=load_csv(train_data_path)
test_data=load_csv(test_data_path)
logging.info("data loaded in training pipeline")

# getting labelled data
X_train = train_data.drop(['LUNG_CANCER'],axis=1)  
X_test=test_data.drop(['LUNG_CANCER'],axis=1) 
y_train = train_data['LUNG_CANCER']
y_test = test_data['LUNG_CANCER']
logging.info("data labelling done")


#model training obj
model_training=ModelTrainer(X_train,X_test,y_train,y_test)
model_training.initiate_model_training()
logging.info("model training done")


