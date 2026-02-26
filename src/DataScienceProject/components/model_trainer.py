import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import mlflow
import dagshub
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.DataScienceProject.exception import CustomException
from src.DataScienceProject.logger import logging
from src.DataScienceProject.utils import save_object, evaluate_models


# Initialize DagsHub MLflow
dagshub.init(
    repo_owner="ayushmishra642001",
    repo_name="DataScience",
    mlflow=True
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array, preprocessor_obj):
        try:
            logging.info("Starting model training")

            os.makedirs("artifacts", exist_ok=True)

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

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
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "subsample": [0.5, 0.7, 1.0],
                    "n_estimators": [100, 200, 300]
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7]
                },
                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.1, 0.01, 0.001],
                    "depth": [3, 5, 7]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.01, 0.001]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]

            if best_model_score < 0.6:
                raise CustomException("No acceptable model found", sys)

            # Create full pipeline
            final_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_obj),
                    ("model", best_model)
                ]
            )

            # Start MLflow experiment
            mlflow.set_experiment("Student Performance Prediction")

            with mlflow.start_run():

                # Evaluate
                predictions = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)

                # Log parameters
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_params(best_model.get_params())

                # Log metrics
                mlflow.log_metrics({
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                })

                # Save model comparison table
                comparison_df = pd.DataFrame({
                    model: {"R2": model_report[model]["score"]}
                    for model in model_report
                }).T

                comparison_path = "artifacts/model_comparison.csv"
                comparison_df.to_csv(comparison_path)
                mlflow.log_artifact(comparison_path)

                # SHAP interpretability (only tree models)
                if best_model_name in [
                    "XGBRegressor",
                    "CatBoosting Regressor",
                    "Random Forest",
                    "Gradient Boosting"
                ]:
                    explainer = shap.Explainer(best_model)
                    shap_values = explainer(X_train)

                    shap.summary_plot(shap_values, X_train, show=False)
                    shap_path = "artifacts/shap_summary.png"
                    plt.savefig(shap_path)
                    plt.close()
                    mlflow.log_artifact(shap_path)

                # Log full pipeline
                mlflow.sklearn.log_model(
                    final_pipeline,
                    name="model"
                )

            # Save full pipeline locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_pipeline
            )

            logging.info(f"Best model: {best_model_name} | R2: {best_model_score}")

            return r2_score(y_test, predictions)

        except Exception as e:
            raise CustomException(e, sys)