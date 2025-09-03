from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.components.data_ingestion import DataIngestionConfig
from src.exception import CustomException
import os
import sys
import pandas as pd
import time
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class model_trainer_config():
    trained_model_path=os.path.join("artifacts","best_model.pkl")
    logging.info("trained model path done")

class ModelTrainer:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.model_trainer_config=model_trainer_config()
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
    def initiate_model_training(self):  
        # Dictionary to hold models and their param grids
        logging.info("entered model training method block")
        models = {
            "MultinomialNB": {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
                    'fit_prior': [True, False]
                }
            },
            "LogisticRegression": {
                'model': LogisticRegression(max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            "RandomForest": {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5]
                }
            },
            "SVM": {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svc', SVC())
                ]),
                'params': {
                    'svc__C': [0.1, 1, 10],
                    'svc__kernel': ['linear', 'rbf']
                }
            },
            "KNN": {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())
                ]),
                'params': {
                    'knn__n_neighbors': [3, 5, 7, 9],
                    'knn__weights': ['uniform', 'distance']
                }
            },
            "DecisionTree": {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10]
                }
            },
            "GradientBoosting": {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5]
                }
            },
            "AdaBoost": {
                'model': AdaBoostClassifier(),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0, 1.5]
                }
            },
            "SGDClassifier": {
                'model': SGDClassifier(max_iter=1000, tol=1e-3),
                'params': {
                    'loss': ['hinge', 'log_loss'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            },
            "MLPClassifier": {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(max_iter=500))
                ]),
                'params': {
                    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'mlp__activation': ['relu', 'tanh'],
                    'mlp__alpha': [0.0001, 0.001]
                }
            }
        }


        # Run grid search for each model
        best_models = {}
        total_start_time=time.time()
        for name, m in models.items():
            try:
                print(f"\nTraining and tuning {name}...")
                model_start_time=time.time()
                grid = GridSearchCV(m['model'], m['params'], cv=5, scoring='accuracy', n_jobs=-1)
                grid.fit(self.X_train, self.y_train)
                model_end_time=time.time()
                logging.info(f"model fitted {name}")
                print(f"Best parameters for {name}: {grid.best_params_}")
                print(f"Best CV accuracy: {grid.best_score_:.4f}")
                

                # Test accuracy
                best_model = grid.best_estimator_
                y_pred = best_model.predict(self.X_test)
                test_acc = accuracy_score(self.y_test, y_pred)
                print(f"Test accuracy: {test_acc:.4f}")
                print(f"time taken {name} {model_end_time-model_start_time}")

                logging.info(f"Best parameters for {name}: {grid.best_params_}")
                logging.info(f"Test accuracy: {test_acc:.4f}")

                best_models[name] = {
                    'model': best_model,
                    'cv_accuracy': grid.best_score_,
                    'test_accuracy': test_acc,
                    'best_params':grid.best_params_
                }
            except Exception as e:
                logging.warning(f"Skipping {name} due to error: {e}")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"\n Total time taken for all models: {total_duration:.2f} seconds")

        # Find model with highest test accuracy
        best_model_name = max(best_models, key=lambda name: best_models[name]['test_accuracy'])
        final_model = best_models[best_model_name]['model']

        try:
            save_object(self.model_trainer_config.trained_model_path,final_model)
            print(f"\n Best model '{best_model_name}' saved to {self.model_trainer_config.trained_model_path}")
            logging.info(f"Best model '{best_model_name}' saved with test accuracy: {best_models[best_model_name]['test_accuracy']:.4f}")
        except Exception as e:
            raise CustomException(e,sys)
          
        # Extract best model information
        best_model_info = {
            "model_name": best_model_name,
            "test_accuracy": best_models[best_model_name]['test_accuracy'],
            "cv_accuracy": best_models[best_model_name]['cv_accuracy'],
            "best_params": str(best_models[best_model_name]['best_params'])  # Convert dict to string for CSV
        }
        try:
            # Convert to DataFrame
            df_best_model = pd.DataFrame([best_model_info])  # Wrap in list to create one-row DataFrame

            # Save to CSV
            csv_path = os.path.join("artifacts", "best_model_info.csv")
            df_best_model.to_csv(csv_path, index=False)

            print(f"Saved best model info to {csv_path}")
            logging.info(f"Saved best model info to {csv_path}")
        except Exception as e:
            raise CustomException(e,sys)



