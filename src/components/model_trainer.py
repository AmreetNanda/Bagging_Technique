import os
import sys
from dataclasses import dataclass

# import all the models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor

from sklearn.metrics import r2_score

from src.Exception import CustomException
from src.Logger import logging
from src.Utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split the training and testing input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "XGBRegressor": XGBRFRegressor()
            }

            # Hyperparameter Tuning
            params = {
                
                "LinearRegression":{},

                "Lasso":{},

                "Ridge":{},

                "Elasticnet":{},

                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },

                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse','absolute_error','poisson'],
                    'max_features':['sqrt','log2'],
                    'n_estimators': [8,16,32,62,128,256]
                },

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
            }

            model_report:dict = evaluate_model(X_train = X_train, 
                                                y_train = y_train,
                                                X_test = X_test,
                                                y_test = y_test,
                                                models = models,
                                                param=params) 
            
            best_model_score = max(sorted(model_report.values())) # To get the best model score from dict
            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score) 
            ]
            best_model = models[best_model_name]
            save_object(file_path = self.model_trainer_config.trained_model_file_path, obj= best_model)

            predicted = best_model.predict(X_test)
            r2_value = r2_score(y_test, predicted)
            return r2_value
        
        except Exception as e:
            raise CustomException (e,sys)