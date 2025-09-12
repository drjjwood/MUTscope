import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix
from classifiers import initialize_classifier
from base_evaluation import extract_data, plot_confusion_matrix, save_predictions, log_custom_wbplots, save_metrics_to_csv, generate_metrics

main_logger = logging.getLogger(__name__)


def load_classical_model(model_path):
    """
    Loads a trained classical machine learning model from a pickle file.

    Args:
        model_path (str): Path to the pickled model file.

    Returns:
        model: The deserialized scikit-learn model object.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    main_logger.info(f"Model loaded from {model_path}")
    return model

def evaluate_classical_model(model, dataloader, config, split_name="validation", logger=None):
    """
    Evaluates a trained classical model (e.g., logistic regression, SVM) on a provided dataset.

    Args:
        model: Trained scikit-learn-compatible model with `predict_proba` or `decision_function`.
        dataloader (DataLoader): DataLoader yielding features and labels.
        config: Configuration object with model settings (e.g., decision threshold).
        split_name (str): Label for this evaluation split (e.g., "train", "validation", "test").
        logger: Optional logger object for tracking metrics (e.g., wandb logger).

    Returns:
        metrics (dict): Dictionary of evaluation metrics (acc, f1, precision, recall, MCC).
        preds (np.ndarray): Binary predictions (0 or 1).
        probs (np.ndarray): Model-predicted probabilities.
        y_true (np.ndarray): Ground truth labels.
        features (np.ndarray): Input features used for prediction.
        target_ids (List): IDs for each sample (e.g., mutation ID).
        cm (np.ndarray): Confusion matrix for the split.
    """
    main_logger.info(f"Evaluating classical model on {split_name} set...")
    _, _, features, labels, target_ids = extract_data(dataloader, outputs=None, include_labels=True)
    X = features
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        logits = model.decision_function(X)
        probs = 1 / (1 + np.exp(-logits))
    
    threshold = config.model.threshold
    main_logger.debug(f"Threshold used for classification: {threshold}")
    preds = (probs > threshold).astype(int)
    y_true = labels
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    else:
        y_true = np.asarray(y_true)
    if hasattr(preds, "cpu"):
        preds = preds.cpu().numpy()
    else:
        preds = np.asarray(preds)

    metrics = generate_metrics(y_true, preds, split_name)
    save_metrics_to_csv(metrics, split_name, config)

    if logger:
        for key, value in metrics.items():
            main_logger.info(f"Logging scalar metric to external logger: {key}: {value}")
            logger.log_metrics({key: value})
    
    cm = confusion_matrix(y_true, preds)
    
    return metrics, preds, probs, y_true, features, target_ids, cm
    

def evaluate_classical_dataset(train_loader, val_loader, test_loader, config, model_path, logger):
    """
    Runs full evaluation pipeline for a classical model across train, validation, and test splits.

    For each split:
    - Loads the model (once)
    - Computes metrics
    - Logs metrics
    - Saves confusion matrix plots
    - Saves prediction outputs
    - Optionally logs W&B plots

    Args:
        train_loader (DataLoader): Dataloader for the training set.
        val_loader (DataLoader): Dataloader for the validation set.
        test_loader (DataLoader): Dataloader for the test set.
        config: Configuration object with model, path, and logging settings.
        model_path (str): Path to the trained classical model (.pkl).
        logger: Logger object for tracking metrics (e.g., wandb logger).

    Returns:
        None. All results are saved or logged externally.
    """
    main_logger.info(f"Running classical evaluation for model at: {model_path}")
    model = load_classical_model(model_path)
    
    main_logger.info("Starting training set evaluation...")
    train_metrics, train_preds, train_probs, train_labels, train_features, train_target_ids, train_cm = evaluate_classical_model(
        model, train_loader, config, split_name="train", logger=logger
    )
    plot_confusion_matrix(train_cm, "train", config)
    save_predictions("train", train_features, train_probs, train_preds, train_target_ids, config, train_labels)

    main_logger.info("Starting validation set evaluation...")
    val_metrics, val_preds, val_probs, val_labels, val_features, val_target_ids, val_cm = evaluate_classical_model(
        model, val_loader, config, split_name="validation", logger=logger
    )
    plot_confusion_matrix(val_cm, "validation", config)
    save_predictions("validation", val_features, val_probs, val_preds, val_target_ids, config, val_labels)

    main_logger.info("Starting test set evaluation...")
    test_metrics, test_preds, test_probs, test_labels, test_features, test_target_ids, test_cm = evaluate_classical_model(
        model, test_loader, config, split_name="test", logger=logger
    )
    plot_confusion_matrix(test_cm, "test", config)
    save_predictions("test", test_features, test_probs, test_preds, test_target_ids, config, test_labels)
    
    if config.settings.track_wb and logger is not None:
        log_custom_wbplots("train", train_labels, train_probs)
        log_custom_wbplots("validation", val_labels, val_probs)
        log_custom_wbplots("test", test_labels, test_probs)
