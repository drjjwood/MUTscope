import os
import pandas as pd
import logging
import json
import yaml
from tqdm import tqdm
from base_embeddings import load_encoded_dataset

main_logger = logging.getLogger(__name__)

def save_config_copy(config):
    """
    For every new experiment, save a snapshot of the config for records.
    Writes to:
        <config.dataset.save_path>/<experiment_name>/config_copy.yaml
    """
    save_dir = os.path.join(
        config.dataset.save_path,
        config.settings.experiment_name
    )
    os.makedirs(save_dir, exist_ok=True)

    config_dict = json.loads(json.dumps(config))

    out_path = os.path.join(save_dir, "config_copy.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(config_dict, f)

    main_logger.info(f"Saved config to {out_path}")

def create_data_paths(config, extension):
    """
    Create file paths for train, validation, and test datasets based on configuration.

    Args:
        config: Configuration object with dataset input path and file names.
        extension (str): File extension (e.g., '.csv' or '.pt').

    Returns:
        tuple: (train_path, val_path, test_path)
    """
    train_path = os.path.join(config.dataset.input_path, f"{config.dataset.train_file_name}{extension}")
    val_path = os.path.join(config.dataset.input_path, f"{config.dataset.val_file_name}{extension}")
    test_path = os.path.join(config.dataset.input_path, f"{config.dataset.test_file_name}{extension}")
    
    return train_path, val_path, test_path 

def get_encoded_dataset(config):
    """
    Load precomputed embeddings for train, validation, and test datasets.

    Args:
        config: Configuration object with dataset paths and names.

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    train_path, val_path, test_path = create_data_paths(config, '.pt')
    train_data = load_encoded_dataset(train_path, data_split = "train")
    val_data = load_encoded_dataset(val_path, data_split = "val")
    test_data = load_encoded_dataset(test_path, data_split = "test")

    return train_data, val_data, test_data

def load_from_csv(train_path, val_path, test_path):  
    """
    Load train, validation, and test CSV files into DataFrames.

    Args:
        train_path (str): File path for the training data.
        val_path (str): File path for the validation data.
        test_path (str): File path for the test data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    main_logger.info(f"Loaded {len(train_df)} rows from train CSV")
    main_logger.info(f"Loaded {len(val_df)} rows from val CSV")
    main_logger.info(f"Loaded {len(test_df)} rows from test CSV")

    return train_df, val_df, test_df
