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
    report = {}

    for model_name, model in models.items():
        param_grid = params.get(model_name, {})

        if param_grid:
            gs = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        predicted = best_model.predict(X_test)
        r2 = r2_score(y_test, predicted)

        report[model_name] = {
            "model": best_model,
            "score": r2
        }

    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)