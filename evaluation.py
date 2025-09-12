import os
import shutil
import logging
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import wandb
from sklearn.metrics import confusion_matrix
from base_model import LitSequenceClassifier, LossTrackingCallback
from classifiers import initialize_classifier
from base_evaluation import extract_data, plot_confusion_matrix, save_predictions, log_custom_wbplots, save_metrics_to_csv, generate_metrics
main_logger = logging.getLogger(__name__)


def evaluate_dataset(trainer, train_loader_eval, val_loader, test_loader, checkpoint_callback, config, embedding_size):
    """
    Evaluate the best model on a given dataset by computing metrics,
    logging visualizations to WandB, and saving predictions.
    Uses a single prediction pass to ensure consistency.
    """
    dataloaders = {
        "train":      train_loader_eval,
        "validation": val_loader,
        "test":       test_loader
    }

    main_logger.info(">>> [evaluate_dataset] loading best checkpoint…")
    model = load_best_checkpoint(trainer, checkpoint_callback, config, embedding_size)
    model.eval()

    for dataset_name, dataloader in dataloaders.items():
        main_logger.debug(f" Calling trainer.predict on dataloader = {dataloader}  len = {len(dataloader)} batches")
        outputs = trainer.predict(model, dataloaders=dataloader)
        main_logger.debug(f"trainer.predict returned {len(outputs)} output batches")
        all_probs, all_logits, all_inputs, all_labels, all_target_ids = extract_data(
            dataloader,
            outputs,
            include_labels=True
        )

        y_true = np.asarray(all_labels, dtype=int)
        y_pred = np.asarray((all_probs > config.model.threshold).int().tolist())

        sanity_check(all_probs, all_labels, all_inputs, all_target_ids, all_logits, y_true, y_pred)

        metrics = generate_metrics(y_true, y_pred, dataset_name, trainer=trainer, config=config)
        save_metrics_to_csv(metrics, dataset_name, config)
        
        save_predictions(dataset_name, all_inputs, all_probs, list(y_pred), all_target_ids, config, all_labels)
        conf_matrix = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(conf_matrix, dataset_name, config)
        if config.settings.track_wb:
            log_custom_wbplots(dataset_name, all_labels, all_probs)


def sanity_check(all_probs, all_labels, all_inputs, all_target_ids, all_logits, y_true, y_pred):
    """
    Perform sanity checks on shapes, values, and label consistency.
    Logs shape information, previews of data, and distribution of labels.
    Raises an assertion if unexpected labels are found.
    """
    # Shape diagnostics
    main_logger.info("→ Sanity check after extract_data:")
    main_logger.info(f"  all_probs  shape : {getattr(all_probs, 'shape', None)}")
    main_logger.info(f"  all_logits shape : {getattr(all_logits, 'shape', None)}")
    main_logger.info(f"  all_inputs shape : {getattr(all_inputs, 'shape', None)}")
    main_logger.info(f"  all_labels shape : {getattr(all_labels, 'shape', None)}")
    main_logger.info(f"  target_ids count : {len(all_target_ids)}")

    # Preview of data
    main_logger.debug(f"  First 5 probs  : {all_probs[:5]}")
    main_logger.debug(f"  First 5 labels : {all_labels[:5]}")
    main_logger.debug(f"  First 5 inputs : {all_inputs[:5]}")
    main_logger.debug(f"  First 5 IDs    : {all_target_ids[:5]}")

    # Shape consistency check
    if all_probs.shape[0] != all_labels.shape[0]:
        main_logger.warning(f"⚠️ MISMATCH: {all_probs.shape[0]} probs vs {all_labels.shape[0]} labels")
    else:
        main_logger.info(f"✔️ Count match: {all_probs.shape[0]}")

    # Label distribution + check for unexpected labels
    unique, counts = np.unique(y_true, return_counts=True)
    label_dist = dict(zip(unique, counts))
    main_logger.info(f"  Label distribution: {label_dist}")
    assert set(unique) <= {0, 1}, f"Unexpected labels found: {unique}"

    # Show a sample of true vs predicted
    main_logger.info(f"  y_true[:10]: {y_true[:10]}")
    main_logger.info(f"  y_pred[:10]: {y_pred[:10]}")

    
def load_best_checkpoint(trainer, checkpoint_callback, config, embedding_size):
    """
    Load the best model checkpoint based on the training strategy.
    Also prints and logs the epoch number and hyperparameters of the best model.
    Then makes a copy of the checkpoint file (or directory, for DeepSpeed)
    into:
        <config.dataset.save_path>/<config.settings.experiment_name>/
    preserving the original file or directory name.
    """
    if config.trainer.strategy == 'deepspeed':
        main_logger.info("Using DeepSpeed to load the best model checkpoint.")
       	checkpoint_path = "./deepspeed_checkpoints/best_model_fp32.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expected checkpoint file not found: {checkpoint_path}")
    
        main_logger.info(f"Loading standard PyTorch checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        best_epoch = checkpoint.get("epoch", "Unknown")

        classifier = initialize_classifier(
            classifier_type=config.model.classifier_head,
            embedding_size=embedding_size,
            config=config
        )
        model = LitSequenceClassifier(classifier, threshold=config.model.threshold)
        model.load_state_dict(state_dict)
        best_path = checkpoint_path
    else:
        main_logger.info(f"Loading best model from checkpoint: {checkpoint_callback.best_model_path}")
        checkpoint_path = checkpoint_callback.best_model_path
        model = LitSequenceClassifier.load_from_checkpoint(checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            best_epoch = checkpoint.get("epoch", "Unknown")
        except Exception as e:
            main_logger.info(f"Could not load checkpoint metadata: {e}")
            best_epoch = "Unknown"
        best_path = checkpoint_path

    log_hyperparameters(model, best_epoch, best_path)
    save_checkpoint_copy(config, best_path)
    return model

    
def log_hyperparameters(model, best_epoch, best_path):
    """
    Log model hyperparameters and checkpoint metadata to W&B.

    Args:
        model: Trained Lightning model.
        best_epoch (int or str): Epoch number of best checkpoint.
        best_path (str): Path to checkpoint.
    """
    main_logger.info(f"Best checkpoint is from epoch: {best_epoch}")
    wandb.log({"best_model_epoch": best_epoch})
    main_logger.info(f"Best model path/directory: {best_path}")
    wandb.log({"best_model_path": best_path})
    wandb.run.summary["best_model_path"] = best_path
    if hasattr(model, "hparams"):
        main_logger.info("Model hyperparameters:")
        wandb.log({"model_hyperparameters": dict(model.hparams)})
        for k, v in model.hparams.items():
            main_logger.info(f"  {k}: {v}")
    else:
    	main_logger.info("No hyperparameters found in the model.")


def save_checkpoint_copy(config, best_path):
    """
    Copy the best checkpoint file to the experiment folder.

    Args:
        config: Configuration object with dataset.save_path and settings.experiment_name.
        best_path (str): Path to the best model checkpoint.
    """
    base_save = getattr(config.dataset, "save_path", ".")
    exp_folder = os.path.join(base_save, config.settings.experiment_name)
    os.makedirs(exp_folder, exist_ok=True)

    if config.trainer.strategy == 'deepspeed':
        dest_file = os.path.join(exp_folder, os.path.basename(best_path))
        shutil.copy(best_path, dest_file)
        main_logger.info(f"Copied DeepSpeed checkpoint file to {dest_file}")
    else:
        src_file = best_path
        dest_file = os.path.join(exp_folder, os.path.basename(src_file))
        shutil.copy(src_file, dest_file)
        main_logger.info(f"Copied checkpoint file to {dest_file}")


def load_lit_model(lit_model, predict_config):
    """
    Load weights into a LitSequenceClassifier from a saved checkpoint.

    Args:
        lit_model (LitSequenceClassifier): An uninitialized Lightning model.
        predict_config: Configuration object with .model.load_path

    Returns:
        LitSequenceClassifier: Model with loaded weights and moved to appropriate device.
    """
    checkpoint = torch.load(predict_config.model.load_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    lit_model.load_state_dict(state_dict)
    lit_model.eval()
    lit_model = lit_model.to("cuda" if torch.cuda.is_available() else "cpu")
    return lit_model

    
def run_predictions(predict_config, predict_loader):
    """
    Run prediction on a dataset using a trained model.

    Args:
        predict_config: Configuration with model path and threshold.
        predict_loader: DataLoader for inference.

    Returns:
        torch.Tensor: Concatenated model predictions across all batches.
    """
    classifier = initialize_classifier(
        classifier_type=predict_config.model.classifier_head,
        embedding_size=predict_config.model.embedding_size,
        config=predict_config,
    )
    lit_base = LitSequenceClassifier(classifier, threshold=predict_config.model.threshold)
    lit_model = load_lit_model(lit_base, predict_config)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    outputs = trainer.predict(model=lit_model, dataloaders=predict_loader)
    all_probs, all_logits, all_inputs, all_labels, all_target_ids = extract_data(
        predict_loader, outputs, include_labels=False
    )
    y_pred = (all_probs > predict_config.model.threshold).int().tolist()

    save_predictions("predict", all_inputs, all_probs, y_pred, all_target_ids, predict_config)

    return torch.cat([out["preds"] for out in outputs], dim=0)
