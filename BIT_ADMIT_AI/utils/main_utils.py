import os
import numpy as np
import dill
import yaml
from pandas import DataFrame

from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.utils.data_generator import generate_dataset as _generate_dataset


def generate_dataset() -> DataFrame:
    "Generates the syntetic dataset"
    try:
        return _generate_dataset()
    except Exception as e:
        logging.error(f"Error occured - {e}")
        raise BitAdmitAIException(e)


def generate_project_template() -> None:
    "Generate a project template"
    pass


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logging.error(f"Error occured - {e}")
        raise BitAdmitAIException(e)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        logging.error(f"Error occured - {e}")
        raise BitAdmitAIException(e)


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        logging.error(f"Error occured - {e}")
        raise BitAdmitAIException(e)


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(os.path.abspath(file_path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array, allow_pickle=False)
    except Exception as e:
        raise BitAdmitAIException(e)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=False)
    except Exception as e:
        raise BitAdmitAIException(e)


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        dir_path = os.path.dirname(os.path.abspath(file_path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise BitAdmitAIException(e)


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop the columns from a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns method of utils")

    try:
        df = df.drop(columns=cols, axis=1, errors="ignore")

        logging.info("Exited the drop_columns method of utils")

        return df
    except Exception as e:
        raise BitAdmitAIException(e)
