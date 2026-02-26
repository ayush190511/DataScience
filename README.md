# End to End Data Science Project


## MLFlow Tracking
import dagshub
dagshub.init(repo_owner='ayushmishra642001', repo_name='DataScience', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)