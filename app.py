from src.DataScienceProject.logger import logging
from src.DataScienceProject.exception import CustomException
from src.DataScienceProject.components.data_ingestion import DataIngestion
from src.DataScienceProject.components.data_ingestion import DataIngestionConfig
from src.DataScienceProject.components.data_transformation import DataTransformation, DataTransformationConfig
import sys 

if __name__ == "__main__":
    logging.info("This is a log message from the main block.")
    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        logging.info("Custom exception occurred.")
        raise CustomException(str(e), sys)
