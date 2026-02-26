import os
import sys
from src.DataScienceProject.exception import CustomException
from src.DataScienceProject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from tqdm import tqdm
import time

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")

    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established: \n{mydb}")
        df = pd.read_sql_query("Select * from student", mydb)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name in tqdm(models, desc="Training models"):
            model = models[model_name]
            param_grid = params[model_name]

            start_time = time.time()
            logging.info(f"Training started for {model_name}")

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "model": best_model,
                "score": test_score
            }

            end_time = time.time()
            logging.info(
                f"{model_name} completed in {end_time - start_time:.2f} seconds "
                f"with R2 score {test_score:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)