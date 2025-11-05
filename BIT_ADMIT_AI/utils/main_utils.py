"""
Main utilities for BIT_ADMIT_AI.

Provides:
- dataset generation wrapper,
- YAML read/write,
- dill save/load,
- NumPy array save/load,
- small DataFrame helpers.

All public helpers raise BitAdmitAIException on failure.
"""

import os
import numpy as np
import dill
import yaml
from pandas import DataFrame

from BIT_ADMIT_AI.logger import logging
from BIT_ADMIT_AI.exceptions import BitAdmitAIException
from BIT_ADMIT_AI.utils.data_generator import generate_dataset as _generate_dataset


def generate_dataset() -> DataFrame:
    """Generate the synthetic admissions dataset.

    Wraps BIT_ADMIT_AI.utils.data_generator.generate_dataset.

    Returns:
        pandas.DataFrame: Generated dataset.

    Raises:
        BitAdmitAIException: If generation fails.
    """
    try:
        return _generate_dataset()
    except Exception as e:
        logging.error(f"Error occured - {e}")
        raise BitAdmitAIException(e)


def generate_project_template() -> None:
    "Generate a project template, for future :)"
    pass


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        BitAdmitAIException: On IO or YAML parse errors.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logging.error(f"Error occured - {e}")
        raise BitAdmitAIException(e)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """Write content to a YAML file.

    Creates parent directories as needed. Optionally replaces an existing file.

    Args:
        file_path: Output path.
        content: Serializable content to dump via yaml.dump.
        replace: If True, remove an existing file before writing.

    Raises:
        BitAdmitAIException: On IO errors.
    """
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
    """Load a Python object serialized with dill.

    Args:
        file_path: Path to the dill file.

    Returns:
        object: Deserialized object.

    Raises:
        BitAdmitAIException: On IO or deserialization errors.
    """
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
    """Persist a NumPy array (.npy).

    Args:
        file_path: Destination path.
        array: Array to save.

    Raises:
        BitAdmitAIException: On IO errors.
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
    """Load a NumPy array (.npy).

    Args:
        file_path: Path to the .npy file.

    Returns:
        np.ndarray: Loaded array.

    Raises:
        BitAdmitAIException: If the file is missing or load fails.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=False)
    except Exception as e:
        raise BitAdmitAIException(e)


def save_object(file_path: str, obj: object) -> None:
    """Serialize a Python object with dill.

    Args:
        file_path: Destination path.
        obj: Python object to serialize.

    Raises:
        BitAdmitAIException: On IO or serialization errors.
    """
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
    """Drop columns from a DataFrame.

    Args:
        df: Input DataFrame.
        cols: Column names to drop.

    Returns:
        pandas.DataFrame: DataFrame without the requested columns.

    Raises:
        BitAdmitAIException: If dropping fails.
    """
    logging.info("Entered drop_columns method of utils")

    try:
        df = df.drop(columns=cols, axis=1, errors="ignore")

        logging.info("Exited the drop_columns method of utils")

        return df
    except Exception as e:
        raise BitAdmitAIException(e)
