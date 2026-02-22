import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "DataScienceProject"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for file in list_of_files:
    file_path = Path(file)
    file_dir = file_path.parent

    if not file_dir.exists():
        logging.info(f"Creating directory: {file_dir}")
        file_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        """
        What “touch” does (conceptually)
          ✅ If the file does not exist → create an empty file
          ✅ If the file already exists → update its modified time
        """
        logging.info(f"Creating file: {file_path}")
        file_path.touch()
    else:
        logging.info(f"File already exists: {file_path}")