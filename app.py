from src.DataScienceProject.logger import logging
from src.DataScienceProject.exception import CustomException
from src.DataScienceProject.components.data_ingestion import DataIngestion
from src.DataScienceProject.components.data_ingestion import DataIngestionConfig
import sys 

if __name__ == "__main__":
    logging.info("This is a log message from the main block.")
    try:
        data_ingestion = DataIngestion()
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion.initiate_data_ingestion()



    except Exception as e:
        logging.info("Custom exception occurred.")
        raise CustomException(str(e), sys)
