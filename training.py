import os
import traceback
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
from base_model import LitSequenceClassifier, LossTrackingCallback
from pytorch_lightning.loggers import WandbLogger
from classifiers import initialize_classifier
from evaluation import evaluate_dataset
from training_classical import train_classical_pipeline
from training_utils import setup_logger, get_model_save_path
import logging
main_logger = logging.getLogger(__name__)

DEEPSPEED_AVAILABLE = False
try:
    from deepspeed import DeepSpeedEngine
    from pytorch_lightning.strategies import DeepSpeedStrategy
    DEEPSPEED_AVAILABLE = True
except ImportError:
    print("DeepSpeed is not available. Skipping DeepSpeed configuration.")


def plot_loss(callback, config):
    """
    Plot training and validation losses over epochs and save the figure.

    Args:
        callback: Instance of `LossTrackingCallback` containing tracked losses.
        config: Configuration object with paths/settings.
    """
    train_losses = callback.train_losses
    val_losses = callback.val_losses
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(config.dataset.save_path,config.settings.experiment_name,
                             f'{config.dataset.train_file_name}_loss.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()

def setup_callbacks(config):
    """
    Set up training callbacks including early stopping, LR monitor, and checkpointing.
    
    Args:
        config: Configuration object with trainer settings.
    
    Returns:
        Tuple of (EarlyStopping, LearningRateMonitor, ModelCheckpoint, LossTrackingCallback)
    """
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=True)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    loss_callback = LossTrackingCallback()
    
    if config.trainer.strategy == 'deepspeed':    
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=None,  # Disable PyTorch Lightning checkpointing
            save_top_k=0,  # Do not save checkpoints
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="{epoch}-{val_loss:.2f}")
    
    return early_stopping, lr_monitor, checkpoint_callback, loss_callback


def setup_strategy(config):
    """
    Set up and return the training strategy based on the configuration.

    If the configuration specifies the 'deepspeed' strategy, the function attempts to create a 
    DeepSpeedStrategy instance. If DeepSpeed is not available or not configured, it falls back 
    to the default strategy.

    Parameters:
        config: Configuration object containing trainer strategy settings.

    Returns:
    strategy (pl.strategies.Strategy or None)

    """
    strategy = config.trainer.strategy
    if config.trainer.strategy == 'deepspeed':
        try:
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(config="configs/ds_config.json", offload_optimizer=True)
        except ImportError:
            main_logger.exception("DeepSpeedStrategy is not available. Falling back to default strategy.")
            strategy = None
    return strategy

def train_pipeline(train_loader, val_loader, test_loader, classifier, config, embedding_size):
    """
    Train and evaluate a PyTorch Lightning model using the specified classifier.
    
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        classifier: Initialized classifier (nn.Module).
        config: Configuration object.
    
    Performs training, saves checkpoints, evaluates on test set, and logs to W&B if enabled.
    """

    main_logger.info(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    lit_model = LitSequenceClassifier(classifier, threshold = config.model.threshold)
    early_stopping, lr_monitor, checkpoint_callback, loss_callback = setup_callbacks(config)
    logger = setup_logger(config)
    strategy = setup_strategy(config)
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[early_stopping, lr_monitor, checkpoint_callback, loss_callback],
        log_every_n_steps=config.trainer.log_every_n_steps,
        fast_dev_run=config.trainer.fast_dev_run,
        profiler=config.trainer.profiler,
        logger=logger,
        precision=config.trainer.precision,
        strategy=strategy
    )
    main_logger.info(f"Using strategy: {trainer.strategy}")
    
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    if config.trainer.strategy == 'deepspeed':
        main_logger.info(f"Saving DeepSpeed checkpoint to './deepspeed_checkpoints'")
        trainer.strategy.deepspeed_engine.save_checkpoint("./deepspeed_checkpoints", tag="final")
    
    trainer.test(model=lit_model, dataloaders=test_loader, verbose=False)
    
    plot_loss(loss_callback, config)        
    evaluate_dataset(trainer, train_loader, val_loader, test_loader, checkpoint_callback, config, embedding_size)


def train_sweep_iteration(train_loader, val_loader, test_loader, embedding_size, config):
    """
    Train the model for one W&B sweep iteration using the current sweep config.
    
    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for test.
        embedding_size: Size of the input embeddings.
        config: Configuration object.
    
    Wraps W&B init block, sets up Lightning model/trainer, and evaluates.
    """
    try:
        with wandb.init(settings=wandb.Settings(start_method="thread")) as run:
            sweep_config = wandb.config 
            classifier = initialize_classifier(
                classifier_type = config.model.classifier_head,
                embedding_size=embedding_size,
                config = config
            )
            lit_model = LitSequenceClassifier(classifier, threshold = config.model.threshold)
            early_stopping, lr_monitor, checkpoint_callback, loss_callback = setup_callbacks(config)
            wandb_logger = WandbLogger(
                project=config.settings.project_name,
                name=f"sweep_run_{run.id}",
            )
            strategy = setup_strategy(config)
            trainer = pl.Trainer(
                max_epochs=sweep_config.max_epochs,
                callbacks=[early_stopping, lr_monitor, checkpoint_callback, loss_callback],
                log_every_n_steps=config.trainer.log_every_n_steps,
                logger=wandb_logger,
                strategy=strategy,
            )
            main_logger.info(f"Using strategy: {trainer.strategy}")

            trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            if config.trainer.strategy == 'deepspeed':
                main_logger.info(f"Saving DeepSpeed checkpoint to './deepspeed_checkpoints'")
                trainer.strategy.deepspeed_engine.save_checkpoint(
                    "./deepspeed_checkpoints", tag=f"final_{run.id}"
                )

            trainer.test(lit_model, dataloaders=test_loader)

            evaluate_dataset(trainer, train_loader, val_loader, test_loader, checkpoint_callback, config, embedding_size)
    except Exception as e:
        main_logger.exception(f"💥 Exception during sweep iteration: {e}")
        traceback.print_exc()

def run_sweep(train_loader, val_loader, test_loader, embedding_size, config):
    """
    Execute a W&B hyperparameter sweep with the provided sweep config.
    
    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for test.
        embedding_size: Size of the input embeddings.
        config: Configuration object with sweep parameters.
    """
    sweep_config = config.sweep
    sweep_id = wandb.sweep(sweep_config, project=config.settings.project_name)

    wandb.agent(
        sweep_id,
        function=lambda: train_sweep_iteration(
            train_loader, val_loader, test_loader, embedding_size, config
        ),count=config.sweep.run_cap,
    )

def run_modelling(train_loader, val_loader, test_loader, config, device):
    """
    Run the full modelling pipeline: either a deep learning model or classical model.
    
    If config.settings.run_sweep is True → launches W&B hyperparameter sweep.
    If config.model.classifier_head is a classical model → runs classical pipeline.
    Else → runs standard Lightning model training.
    
    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for test.
        config: Configuration object.
        device: Device to use for training (CPU or GPU).
    """
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'x') and hasattr(train_dataset, 'logits'):
        embedding_size = train_dataset.x.shape[1] + 1  # Add logit size, better way to do this?
        print(f"Embedding size (x + logits): {embedding_size}")
    else:
        raise ValueError("The dataset does not have embeddings (`x`) or logits.")
    main_logger.info(f"Embedding size: {embedding_size}")

    if config.settings.run_sweep:
        main_logger.info("Running hyperparameter sweep...")
        run_sweep(train_loader, val_loader, test_loader, embedding_size, config)
    else:
        if config.model.classifier_head in ['LR', 'SVC', 'GBC','BRF']:
            main_logger.info("Running classical training pipeline...")
            train_classical_pipeline(train_loader, val_loader, test_loader, config)
        else:
            main_logger.info("Running standard training pipeline...")
            classifier = initialize_classifier(
                classifier_type=config.model.classifier_head,
                embedding_size=embedding_size,
                config=config,
            )
            train_pipeline(train_loader, val_loader, test_loader, classifier, config, embedding_size)
