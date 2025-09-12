import wandb
import os
from pytorch_lightning.loggers import WandbLogger

def get_model_save_path(config, classifier_type):
    """
    Construct the full path for saving a classical model.
    
    - Base dir: config.model.save_path or "./models"
    - Subdir: config.settings.experiment_name
    - Filename: "{classifier_type}_model.pkl"
    
    Ensures the directory exists.
    
    Returns:
        str: Full path to save the model.
    """
    base_dir = getattr(config.model, "save_path", "./models")
    save_dir = os.path.join(base_dir, config.settings.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{classifier_type}_model.pkl")


def setup_logger(config):
    """
    Set up and return the logger based on configuration settings.

    If W&B tracking is enabled (config.settings.track_wb is True), this function logs in to W&B 
    and creates a WandbLogger. Otherwise, it returns None.

    Parameters:
        config: Configuration object containing settings for tracking and project names.

    Returns:
        logger (WandbLogger or None): The configured logger instance or None if W&B tracking is disabled.
    """
    if config.settings.track_wb:
        wandb.login(key=config.settings.wandb_key)
        wandb_logger = WandbLogger(
            project=config.settings.project_name,
            name=config.settings.experiment_name,
        )
	logger = wandb_logger
    else:
	logger = None
    return logger
