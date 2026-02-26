import os
import sys
from dataclasses import dataclass

from urllib.parse import urlparse
import numpy as np
import mlflow
import dagshub

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.DataScienceProject.exception import CustomException
from src.DataScienceProject.logger import logging
from src.DataScienceProject.utils import save_object, evaluate_models


dagshub.init(
    repo_owner='ayushmishra642001',
    repo_name='DataScience',
    mlflow=True
)



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return rmse, mae, r2
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2']
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [100, 200, 300]
                },
                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'subsample': [0.5, 0.7, 1.0],
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [100, 200, 300]
                },

                "Linear Regression": {
                    # No hyperparameters to tune for Linear Regression
                },
                "K-Neighbors Regressor": {
                    # 'n_neighbors': [3, 5, 7],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.001],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'depth': [3, 5, 7]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.01, 0.001],
                    # 'loss': ['linear', 'square', 'exponential']
                }
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # best_model_score = max(model_report.values())
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            # best_model = models[best_model_name]   

            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]  

            print("This is the best model name:\n", best_model_name)

            # model_names = list(params.keys())

            # actual_model = ""

            # for model in model_names:
            #     if best_model_name == model:
            #         actual_model = actual_model + model

            # best_params = params[actual_model]

            actual_model = best_model_name
            best_params = best_model.get_params()

            # mlflow.set_registry_url("https://dagshub.com/ayushmishra642001/DataScience.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Ensure acceptable model
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            mlflow.set_experiment("Student Performance Prediction")
            
            with mlflow.start_run():
            
                predicted_qualities = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)
            
                # Log best model name
                mlflow.log_param("best_model", best_model_name)
            
                # Log best parameters selected by GridSearch
                mlflow.log_params(best_model.get_params())
            
                # Log evaluation metrics
                mlflow.log_metrics({
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                })
            
                # Log trained model
                mlflow.sklearn.log_model(best_model, name="model", serialization_format="skops")

            logging.info(f"Best model is {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)