from src.DataScienceProject.logger import logging
from src.DataScienceProject.exception import CustomException
import sys 

if __name__ == "__main__":
    logging.info("This is a log message from the main block.")
    try:
        a = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Custom exception occurred.")
        raise CustomException(str(e), sys)
