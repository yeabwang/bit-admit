import os
from pathlib import Path
import logging

logger = logging.getLogger("__log__")
project_name = "BIT_ADMIT_AI"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configration/__init__.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config.py",
    f"{project_name}/entity/artifact.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training.py",
    f"{project_name}/pipeline/prediction.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/template_creator.py",
    f"{project_name}/utils/data_generator.py",
    f"{project_name}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/scheman.yaml",
    ".gitignore",
    "LICENCE",
    "README.md",
    "dataset/"
]

for pathfile in list_of_files:
    path_name = Path(pathfile)

    dir_name, file_name = os.path.split(path_name)

    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(path_name) or os.path.getsize(path_name) != 0:
        with open(path_name, "w") as file:
            pass
    else:
        logger.warning(f"{path_name}_file alread exists")
        continue
