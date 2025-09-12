import os
import shutil
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pytorch_lightning as pl
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
)

from base_model import LitSequenceClassifier, LossTrackingCallback
from classifiers import initialize_classifier

main_logger = logging.getLogger(__name__)

def generate_metrics(y_true, y_pred, split_name, trainer=None, config=None):
    """
    Compute classification metrics, log them, and return a dictionary
    with prefixed keys like '<split_name>_best_model_acc'.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
        split_name (str): One of "train", "validation", or "test".
        trainer (optional): PyTorch Lightning Trainer object.
        config (optional): Configuration with logging settings.

    Returns:
        dict: Metrics with keys like 'train_best_model_acc', etc.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        f"{split_name}_best_model_acc": acc,
        f"{split_name}_best_model_f1": f1,
        f"{split_name}_best_model_precision": precision,
        f"{split_name}_best_model_recall": recall,
        f"{split_name}_best_model_mcc": mcc,
    }

    main_logger.debug(f"Number of samples evaluated: {len(y_true)}")
    main_logger.info(
        f"{split_name.capitalize()} Metrics: "
        f"Accuracy: {acc:.4f}, F1: {f1:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}"
    )

    if trainer and hasattr(trainer, "logger"):
        for key, value in metrics.items():
            main_logger.info(f"Logging scalar metric: {key}: {value}")
            trainer.logger.log_metrics({key: value})

    return metrics


def save_metrics_to_csv(metrics_dict, dataset_name, config):
    """
    Save evaluation metrics to a CSV file in the experiment directory.

    Args:
        metrics_dict (dict): Dictionary of scalar metrics.
        dataset_name (str): One of 'train', 'validation', 'test'.
        config: Config object with save path and experiment name.
    """
    save_dir = os.path.join(config.dataset.save_path, config.settings.experiment_name, "metrics")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, f"{dataset_name}_metrics.csv")
    df = pd.DataFrame([metrics_dict])  # Wrap in list to make a single-row DataFrame
    df.to_csv(csv_path, index=False)
    main_logger.info(f"Saved {dataset_name} metrics to {csv_path}")


def log_custom_wbplots(dataset_name, all_labels, all_probs):
    """
    Log precision-recall and ROC curves to Weights & Biases (W&B).

    Args:
        dataset_name (str): Name of the dataset (e.g. 'train', 'validation', 'test').
        all_labels (list or array): Ground-truth binary labels.
        all_probs (list or array): Predicted probabilities for the positive class.
    """
    wandb.log({
        f"{dataset_name}_pr_curve": wandb.plot.pr_curve(
            np.array(all_labels),
            np.stack([1 - np.array(all_probs), np.array(all_probs)], axis=-1),
            title=f"{dataset_name.capitalize()} Precision-Recall Curve",
        ),
        f"{dataset_name}_roc_curve": wandb.plot.roc_curve(
            np.array(all_labels),
            np.stack([1 - np.array(all_probs), np.array(all_probs)], axis=-1),
            title=f"{dataset_name.capitalize()} ROC Curve",
        ),
    })

def plot_confusion_matrix(conf_matrix, dataset_name, config):
    """
    Plot and save a confusion matrix heatmap for a given dataset.

    Args:
        conf_matrix (array): 2x2 confusion matrix.
        dataset_name (str): Dataset name ('train', 'validation', or 'test').
        config (Config): Configuration object with save paths and filenames.
    """

    save_path = os.path.join(config.dataset.save_path,config.settings.experiment_name)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix for {dataset_name} Dataset")

    dataset_filename_lookup = {
        "train": f'{config.dataset.train_file_name}',
        "validation": f'{config.dataset.val_file_name}',
        "test": f'{config.dataset.test_file_name}'
    }
    file_path = os.path.join(save_path, f"{dataset_filename_lookup[dataset_name]}_confusion_matrix.png")
    plt.savefig(file_path)
    plt.close(fig)
    main_logger.info(f"Confusion matrix plot saved to {file_path}")


def save_predictions(dataset_name, inputs, probabilities, binary_preds, target_ids, config, labels=None):
    """
    Save predictions and metadata to a CSV file in the experiment directory.

    Args:
        dataset_name (str): Dataset name ('train', 'validation', 'test', or 'predict').
        inputs (List or Tensor): Input features (already detached to CPU).
        probabilities (list): Model-predicted probabilities.
        binary_preds (list): Binary predictions (0 or 1).
        target_ids (list): Sequence/sample identifiers.
        config (Config): Experiment configuration object.
        labels (list, optional): True labels (if available).
    """
    output_path = get_prediction_output_path(dataset_name, config)
    df = prepare_prediction_dataframe(inputs, probabilities, binary_preds, target_ids, labels)
    df.to_csv(output_path, index=False)
    main_logger.info(f"Saved predictions to {output_path}")


def get_prediction_output_path(dataset_name, config) -> str:
    """
    Construct the full file path for saving predictions.

    Args:
        dataset_name (str): Name of the dataset ('train', 'validation', etc.).
        config (Config): Config object with dataset save paths and filenames.

    Returns:
        str: Full file path to save the CSV of predictions.
    """
    dataset_filename_lookup = {
        "train":      getattr(config.dataset, "train_file_name", "train.csv"),
        "validation": getattr(config.dataset, "val_file_name",   "validation.csv"),
        "test":       getattr(config.dataset, "test_file_name",  "test.csv"),
        "predict":    getattr(config.dataset, "predict_file_name", "predict.csv"),
    }

    base_dir = getattr(config.dataset, "save_path", ".")
    save_dir = os.path.join(base_dir, config.settings.experiment_name, "predictions")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{dataset_filename_lookup[dataset_name]}_predict.csv"
    return os.path.join(save_dir, filename)


def prepare_prediction_dataframe(inputs, probabilities, binary_preds, target_ids, labels=None) -> pd.DataFrame:
    """
    Format predictions, inputs, and metadata into a pandas DataFrame.

    Args:
        inputs (List or Tensor): Input feature vectors (usually torch tensors).
        probabilities (list): Predicted probabilities.
        binary_preds (list): Predicted binary labels.
        target_ids (list): Sequence or sample identifiers.
        labels (list, optional): Ground-truth labels, if available.

    Returns:
        pd.DataFrame: Structured DataFrame ready to save.
    """
    clean_inputs = [
        [float(val) for val in (x.tolist() if hasattr(x, "tolist") else x)]
        for x in inputs
    ]

    data = {
        "inputs": clean_inputs,
        "probabilities": probabilities,
        "binary_predictions": binary_preds,
        "target_id": target_ids,
    }

    if labels is not None:
        data["labels"] = labels

    return pd.DataFrame(data)


def extract_data(dataloader, outputs=None, include_labels=True):
    """
    Extract and flatten model predictions and inputs from batches.

    Args:
        dataloader (DataLoader): The dataloader used during prediction.
        outputs (List[Dict], optional): Model outputs returned by `trainer.predict`.
        include_labels (bool): Whether to extract labels.

    Returns:
        Tuple: (all_probs, all_logits, all_inputs, all_labels, all_target_ids)
    """
    if outputs is not None:
        return _extract_from_outputs(dataloader, outputs, include_labels)
    else:
        return _extract_from_dataloader_only(dataloader, include_labels)


def _extract_from_outputs(dataloader, outputs, include_labels):
    """
    Internal helper to unpack predictions from model outputs.

    Args:
        dataloader (DataLoader): The data loader for batching input data.
        outputs (List[Dict]): List of model outputs per batch.
        include_labels (bool): Whether to extract and return true labels.

    Returns:
        Tuple: Flattened tensors/lists of predictions, inputs, labels, and IDs.
    """
    all_inputs = []
    all_target_ids = []
    all_probs = []
    all_logits = []
    all_labels = [] if include_labels else None

    main_logger.info(f">>> [extract_data] unpacking {len(outputs)} output batches against {len(dataloader)} data batches")

    for idx, (batch, output) in enumerate(zip(dataloader, outputs)):
        main_logger.info(f"  — batch {idx}:")
        
        # Extract components
        probs = output["preds"].detach().cpu()
        logits = output["logits"].detach().cpu()
        target_ids = output["target_ids"]
        inputs = batch["x_with_logits"].detach().cpu()

        # Logging shapes
        main_logger.info(f"      x_with_logits shape: {tuple(inputs.shape)}")
        main_logger.info(f"      probs shape:         {tuple(probs.shape)}")
        main_logger.info(f"      logits shape:        {tuple(logits.shape)}")

        # Append to results
        all_inputs.append(inputs)
        all_probs.append(probs)
        all_logits.append(logits)

        # Labels
        if include_labels:
            labels = output.get("labels", None)
            if labels is None:
                raise KeyError(f"Missing 'labels' in predict output for batch {idx}")
            labels = labels.detach().cpu()
            all_labels.append(labels)
            main_logger.info(f"      labels shape:       {tuple(labels.shape)}")

        # Target IDs
        count_ids = len(target_ids) if isinstance(target_ids, (list, tuple)) else 1
        all_target_ids.extend(target_ids if isinstance(target_ids, (list, tuple)) else [target_ids])
        main_logger.info(f"      target_ids count:    {count_ids}")

    # Concatenate
    main_logger.info(">>> [extract_data] concatenating …")
    all_inputs = torch.cat(all_inputs, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    if include_labels:
        all_labels = torch.cat(all_labels, dim=0)

    main_logger.info(">>> [extract_data] done.")
    return all_probs, all_logits, all_inputs, all_labels, all_target_ids


def _extract_from_dataloader_only(dataloader, include_labels):
    """
    Internal helper to extract inputs and labels directly from a dataloader.

    Used when model outputs are not available.

    Args:
        dataloader (DataLoader): The dataloader object.
        include_labels (bool): Whether to extract labels.

    Returns:
        Tuple: (None, None, all_inputs, all_labels, all_target_ids)
    """
    all_inputs = []
    all_target_ids = []
    all_labels = [] if include_labels else None

    for batch in dataloader:
        inputs = batch["x_with_logits"].cpu()
        all_inputs.append(inputs)

        if include_labels:
            labels = batch["label"].cpu()
            all_labels.append(labels)

        target_ids = batch["target_id"]
        all_target_ids.extend(target_ids if isinstance(target_ids, (list, tuple)) else [target_ids])

    all_inputs = torch.cat(all_inputs, dim=0)
    if include_labels:
        all_labels = torch.cat(all_labels, dim=0)

    all_probs = None
    all_logits = None

    main_logger.info(">>> [extract_data] done (no outputs).")
    return all_probs, all_logits, all_inputs, all_labels, all_target_ids
