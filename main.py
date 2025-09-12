import yaml
import random
import argparse
import os
import wandb
import torch
import numpy as np
import tensorflow as tf
import json
import logging
from easydict import EasyDict
from dataloaders import create_dataloaders, create_predict_dataloader
from training import run_modelling
from evaluation import run_predictions
from dms_predictor import create_dms_df
from run_interpretation import run_training_interpretation, run_prediction_interpretation
from data_io import save_config_copy
logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)
DEEPSPEED_AVAILABLE = False
try:
    import deepspeed
    os.environ["DEEPSPEED_LOG_LEVEL"] = "debug"
    os.environ["PYTORCH_LIGHTNING_DEBUG"] = "1"
    DEEPSPEED_AVAILABLE = True
except ImportError:
    main_logger.info("DeepSpeed is not available. Skipping DeepSpeed configuration.")


def run_trainer(train_config, device):
    """
    Trains and evaluates a single model instance on the specified train/val/test split.
    This function supports both standalone training and training under a
    Weights & Biases (W&B) sweep. It loads the data, runs model training, 
    and optionally performs interpretation on the trained model.
    """
    safe_config = json.loads(json.dumps(train_config))

    if train_config.settings.run_sweep:
        train_loader, val_loader, test_loader = create_dataloaders(train_config, device)
        run_modelling(train_loader, val_loader, test_loader, train_config, device)

    else:
        run = wandb.init(
        project = train_config.settings.project_name,
        name    = train_config.settings.experiment_name,
        config  = safe_config,
        reinit  = True)

        train_loader, val_loader, test_loader = create_dataloaders(train_config, device)

        log_dataloader_stats("Train", train_loader)
        log_dataloader_stats("Validation", val_loader)
        log_dataloader_stats("Test", test_loader)
        
        run_modelling(train_loader, val_loader, test_loader, train_config, device)
        if train_config.predictions.interpretation:
            run_training_interpretation(train_config)
        run.finish()
        

def run_predictor(predict_config, device):
    """
    Create a data loader for prediction and run inference.
    """
    if predict_config.predictions.dms == True:
        create_dms_df(predict_config)
    predict_loader = create_predict_dataloader(predict_config, device)

    log_dataloader_stats("Prediction", predict_loader)
    
    run_predictions(predict_config, predict_loader)
    
    if predict_config.predictions.interpretation == True:
        run_prediction_interpretation(predict_config)


def log_dataloader_stats(name: str, dataloader):
    dataset_len = len(dataloader.dataset)
    batch_count = len(dataloader)
    batch_size = dataloader.batch_size

    main_logger.info(f"{name} — # Batches: {batch_count}, # Samples: {dataset_len}, Batch Size: {batch_size}")
    
    if hasattr(dataloader, 'drop_last') and dataloader.drop_last:
        dropped = dataset_len % batch_size
        if dropped > 0:
            main_logger.warning(f"{name} — drop_last=True: {dropped} samples will be dropped")

    
def get_device():
    """
    Determines the appropriate device for PyTorch computations based on available hardware.

    The function checks for the availability of CUDA (NVIDIA GPUs) first, then MPS (Apple’s Metal 
    Performance Shaders) for macOS, and falls back to the CPU if neither is available.

    Returns:
        torch.device: The device to be used for tensor computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda") 
    elif torch.backends.mps.is_available():
        return torch.device("mps")  
    else:
        return torch.device("cpu")  

    
def set_all_seeds(seed):
  """
  Sets all seeds for reproducibility.

  Args:
    seed: The integer seed value.
  """
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def main():
    """This function parses command-line arguments, loads the YAML configuration,
    sets the random seed for reproducibility, determines the compute device,
    saves a copy of the configuration, and launches either the training or prediction
    pipeline based on the `--mode` argument.

    Command-line Arguments:
        --mode (-m): str, optional
            The mode to run the pipeline in. Options are 'train' or 'predict'.
            Defaults to 'train'.
        --config_path (-c): str, optional
            Path to the YAML configuration file. Defaults to 'configs/config.yaml'.

    Behavior:
        - Loads and parses the YAML config into an EasyDict object.
        - Sets all seeds (Python, NumPy, PyTorch, etc.).
        - Initializes the device (GPU, MPS, or CPU).
        - Saves a copy of the config for reproducibility.
        - Runs either the training loop or prediction loop depending on mode.
    """
    parser = argparse.ArgumentParser(description='add args for training the model')
    parser.add_argument('--mode', '-m', default='train', type=str,
                    choices=['train', 'predict'],
                    help="Mode to run: 'train' or 'predict'")
    parser.add_argument('--config_path', '-c', default='configs/config.yaml', type=str, 
                        help="Path to configuration YAML file")
    args = parser.parse_args()
    
    device = get_device()
    main_logger.info(f"Using device: {device}")
    config_path = args.config_path 

    try:
        with open(config_path, 'r') as file:
            config = EasyDict(yaml.safe_load(file))
    except Exception as e:
        main_logger.error(f"Error: Unable to load config from {config_path}. Exception: {e}")
        config = None

    if config and config.settings.seed:
        set_all_seeds(config.settings.seed)
    else:
        set_all_seeds(42) 
        
    save_config_copy(config)
        
    if args.mode == 'train':
        run_trainer(config, device)
    elif args.mode == 'predict':
        run_predictor(config, device)
    else:
        main_logger.error('Error, mode should be either train or predict')

if __name__ == "__main__":
    main()
