import logging
import os
from datetime import datetime

LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, LOG_FILE),
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
