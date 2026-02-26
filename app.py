from src.DataScienceProject.logger import logging
from src.DataScienceProject.exception import CustomException
from src.DataScienceProject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.DataScienceProject.components.data_transformation import DataTransformation, DataTransformationConfig
from src.DataScienceProject.components.model_trainer import ModelTrainer, ModelTrainerConfig
import sys 

if __name__ == "__main__":
    logging.info("This is a log message from the main block.")
    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        #Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"R2 Score is {r2_score}")


    except Exception as e:
        logging.info("Custom exception occurred.")
        raise CustomException(str(e), sys)
