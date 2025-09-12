import os
import torch
import pandas as pd
import logging
import random
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from preprocessing import seq_length_process, prepare_input_dfs
from cached_dataloaders import get_encoded_dataset_cached, build_split_dataset_from_cache

main_logger = logging.getLogger(__name__)
SUPPORTED_EMBEDDINGS = ['esm1b', 'esm2']


def random_pad_collate_fn(batch, dataset, desired_batch_size):
    """
    Collate function that pads an incomplete batch by randomly sampling extra examples 
    from the provided dataset to reach the desired batch size.
    
    Args:
        batch: List of examples produced by the dataset.
        dataset: The dataset from which extra samples can be drawn.
        desired_batch_size: The fixed batch size desired.
    
    Returns:
        A collated batch of size `desired_batch_size`.
    """
    current_batch_size = len(batch)
    if current_batch_size < desired_batch_size:
        pad_count = desired_batch_size - current_batch_size
        extra_samples = random.choices(dataset, k=pad_count)
        batch.extend(extra_samples)
    return default_collate(batch)

def create_predict_dataloader(config, device):
    """
    Create a DataLoader for prediction data using the per-ID embedding cache.
    """
    main_logger.info("Processing CSV file...")
    csv_path = os.path.join(
        config.dataset.input_path, f"{config.dataset.predict_file_name}.csv"
    )
    predict_df = pd.read_csv(csv_path)
    df_proc = seq_length_process(predict_df)

    use_cache = not getattr(config.dataset, "disable_cache", False)

    if str(config.settings.embedding_type).lower() not in SUPPORTED_EMBEDDINGS:
        raise ValueError(
            f"Embedding type {config.settings.embedding_type} is not recognized. "
            "Please specify a valid embedding type (e.g., 'esm2')."
        )

    encoded_ds = build_split_dataset_from_cache(
        df=df_proc, device=device, use_cache=use_cache, config=config
    )

    main_logger.info("Creating DataLoader (predict)…")
    predict_loader = DataLoader(
        encoded_ds,
        batch_size=config.dataset.loader_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )
    return predict_loader

    
def create_dataloaders(config, device):
    """
    Load train, validation, and test data as PyTorch DataLoaders.

    Args:
        config: A configuration object or dictionary with run params e.g. dataset paths and batch sizes.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for train, validation, and test sets.
      """
    main_logger.info("Processing CSV files...")
    train_df_raw, val_df_raw, test_df_raw = prepare_input_dfs(config)

    train_df = seq_length_process(train_df_raw)
    val_df   = seq_length_process(val_df_raw)
    test_df  = seq_length_process(test_df_raw)

    main_logger.info(f"Train rows after seq_length_process: {len(train_df)}")
    main_logger.info(f"Val rows after seq_length_process: {len(val_df)}")
    main_logger.info(f"Test rows after seq_length_process: {len(test_df)}")

    if str(config.settings.embedding_type).lower() not in SUPPORTED_EMBEDDINGS:
        raise ValueError(
            f"Embedding type {config.settings.embedding_type} is not recognized. "
            "Please specify a valid embedding type (e.g., 'esm2')."
        )

    use_cache = not getattr(config.dataset, "disable_cache", False)

    # This will generate only missing IDs and then assemble the encoded datasets
    train_dataset, val_dataset, test_dataset = get_encoded_dataset_cached(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        device=device,
        config=config,
        use_cache=use_cache,
    )

    main_logger.info("Creating DataLoaders...")
    collate_fn_train = partial(random_pad_collate_fn, dataset=train_dataset, desired_batch_size=config.dataset.loader_batch_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.loader_batch_size, 
        shuffle=True, 
        num_workers=config.dataset.num_workers, 
        drop_last=False, 
        collate_fn=collate_fn_train
    )
    collate_fn_val = partial(random_pad_collate_fn, dataset=val_dataset, desired_batch_size=config.dataset.loader_batch_size)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.dataset.loader_batch_size, 
        shuffle=False, 
        num_workers=config.dataset.num_workers, 
        drop_last=False, 
        collate_fn=collate_fn_val
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.dataset.loader_batch_size, 
        shuffle=False, 
        num_workers=config.dataset.num_workers, 
        drop_last=True,  # Don't want to duplicate anything in the test set 
    )
    return train_loader, val_loader, test_loader

