import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import Precision, Recall, F1Score, BinaryMatthewsCorrCoef
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection


class LitSequenceClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for binary sequence classification.
    
    This model wraps a classifier (e.g. an MLP) and computes various evaluation metrics.
    """
    def __init__(self, classifier, threshold=0.5):
        """
        Initializes the LitSequenceClassifier.
        
        Args:
            classifier (nn.Module): The classifier model (e.g., an MLP).
            threshold (float): Threshold for converting logits to binary predictions.
        """
        super().__init__()
        self.classifier = classifier
        self.threshold = threshold
        self.save_hyperparameters()
        self.best_val_loss = float("inf")
        self.dataset_name = "test"  # Default dataset name; updated during evaluation.
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') 
        
        self.train_metrics = MetricCollection({
            "train_precision": Precision(task="binary", average='macro', threshold=threshold),
            "train_recall": Recall(task="binary", average='macro', threshold=threshold),
            "train_f1": F1Score(task="binary", average='macro', threshold=threshold),
            "train_mcc": BinaryMatthewsCorrCoef(threshold=threshold),
        })

        self.val_metrics = MetricCollection({
            "val_precision": Precision(task="binary", average='macro', threshold=threshold),
            "val_recall": Recall(task="binary", average='macro', threshold=threshold),
            "val_f1": F1Score(task="binary", average='macro', threshold=threshold),
            "val_mcc": BinaryMatthewsCorrCoef(threshold=threshold),
        })

        self.test_metrics = MetricCollection({
            "test_precision": Precision(task="binary", average='macro', threshold=threshold),
            "test_recall": Recall(task="binary", average='macro', threshold=threshold),
            "test_f1": F1Score(task="binary", average='macro', threshold=threshold),
            "test_mcc": BinaryMatthewsCorrCoef(threshold=threshold),
        })

    def forward(self, embeddings):
        """
        Forward pass through the classifier.
        
        Args:
            embeddings (torch.Tensor): Input embeddings.
            
        Returns:
            torch.Tensor: Logits for binary classification (squeezed).
        """
        target_device = next(self.classifier.parameters()).device
        target_dtype = next(self.classifier.parameters()).dtype
        embeddings = embeddings.to(target_device).to(target_dtype)
        logits = self.classifier(embeddings)
        return logits.squeeze(-1)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch (dict): Batch of data containing 'x_with_logits' and 'label'.
            batch_idx (int): Index of the batch.
            
        Returns:
            torch.Tensor: The training loss.
        """
        labels, logits, predictions, batch_size = self.step_setup(batch)
        loss = self.criterion(logits, labels.float())
        computed = self.train_metrics(predictions, labels)
        self.log_dict(computed, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("train_acc", ((predictions > self.threshold) == labels).float().mean(), prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch (dict): Batch of data containing 'x_with_logits' and 'label'.
            batch_idx (int): Index of the batch.
            
        Returns:
            torch.Tensor: The validation loss.
        """
        labels, logits, predictions, batch_size = self.step_setup(batch)
        loss = self.criterion(logits, labels.float())
        computed = self.val_metrics(predictions, labels)
        self.log_dict(computed, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_acc", ((predictions > self.threshold) == labels).float().mean(), prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Testing step.
        
        Args:
            batch (dict): Batch of data containing 'x_with_logits' and 'label'.
            batch_idx (int): Index of the batch.
            
        Returns:
            torch.Tensor: The test loss.
        """
        labels, logits, predictions, batch_size = self.step_setup(batch)
        loss = self.criterion(logits, labels.float())
        prefix = f"{self.dataset_name}_"
        self.log(f"{prefix}loss", loss, sync_dist=True, batch_size=batch_size)
        self.log(f"{prefix}acc", ((predictions > self.threshold) == labels).float().mean(), sync_dist=True, batch_size=batch_size)
        computed = self.test_metrics(predictions, labels)
        self.log_dict(computed, sync_dist=True, batch_size=batch_size)

        return loss

    def step_setup(self,batch):
        '''Get labels, logits, predictions, batch_size for running each step'''
        embeddings = batch['x_with_logits']
        labels = batch['label']
        logits = self.forward(embeddings)
        predictions = torch.sigmoid(logits)
        batch_size = labels.size(0)
        return labels, logits, predictions, batch_size

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        
        Saves a checkpoint if the current validation loss is the best seen so far.
        """
        print("[on_train_epoch_end] Callback triggered.")
        if "val_loss" not in self.trainer.callback_metrics:
            print("val_loss is not logged. Skipping checkpoint save.")
            return

        current_val_loss = self.trainer.callback_metrics["val_loss"].item()

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            print(f"New best model found with val_loss={current_val_loss}. Saving checkpoint...")
            if isinstance(self.trainer.strategy, pl.strategies.DeepSpeedStrategy):
                print("DeepSpeed strategy detected. Saving checkpoint...")
                try:
                    client_state = {"epoch": self.current_epoch}
                    self.trainer.strategy.deepspeed_engine.save_checkpoint(
                        save_dir="./deepspeed_checkpoints/best",
                        tag="best_model",
                        client_state = {"epoch": self.current_epoch}
                    )
                    print("Checkpoint saved successfully.")
                    
                    torch.save(
                        {
                            "state_dict": self.state_dict(),
                            "epoch": self.current_epoch,
                        },
                        "./deepspeed_checkpoints/best_model_fp32.pt"
                    )
                    print("Full fp32 checkpoint saved to ./deepspeed_checkpoints/best_model_fp32.pt")

                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")

    def configure_optimizers(self):
        """
        Configures and returns the optimizer and learning rate scheduler.
        
        Returns:
            dict: Dictionary with the optimizer and lr_scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )
        if isinstance(self.trainer.strategy, pl.strategies.DeepSpeedStrategy):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=3,
                threshold=1e-4,
                threshold_mode="rel",
                verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for inference. May be called on unlabeled data.
        """
        embeddings = batch['x_with_logits']
        logits = self(embeddings)
        preds = torch.sigmoid(logits)

        # Safely grab labels if they exist
        labels = batch.get("label", None)
        ids = batch["target_id"]

        return {
            "preds":      preds.detach().cpu(),
            "logits":     logits.detach().cpu(),
            "labels":     labels.detach().cpu() if labels is not None else None,
            "target_ids": ids,
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns the state dictionary for checkpointing, excluding non-serializable attributes.
        
        Args:
            destination: (optional) Destination dictionary.
            prefix: (optional) Prefix for the keys.
            keep_vars: (optional) Whether to keep variables.
            
        Returns:
            dict: The state dictionary.
        """
        state = super().state_dict(destination, prefix, keep_vars)
        return {k: v for k, v in state.items() if not isinstance(v, torch._C._distributed_c10d.ProcessGroup)}

    def check_device_consistency(self, batch, model_device):
        """
        Ensures that all tensors in the batch are on the specified device.
        
        Args:
            batch (dict): Batch of data containing tensors.
            model_device (torch.device): The expected device.
            
        Raises:
            AssertionError: If any tensor in the batch is not on the expected device.
        """
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                assert tensor.device == model_device, f"{key} is on {tensor.device}, expected {model_device}!"


class LossTrackingCallback(Callback):
    """
    A PyTorch Lightning Callback for tracking training and validation losses.
    This callback stores the loss values at the end of each training and validation epoch.
    The losses are appended to `train_losses` and `val_losses` lists respectively.
    """

    def __init__(self):
        """
        Initialize the LossTrackingCallback.
        """
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of a training epoch.

        This method retrieves the 'train_loss' from the trainer's callback metrics and
        appends it to the train_losses list.
        """
        if 'train_loss' in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of a validation epoch.

        This method retrieves the 'val_loss' from the trainer's callback metrics and
        appends it to the val_losses list. Losses during sanity checking are ignored.
        """
        if trainer.sanity_checking:
            return  # Ignore losses during sanity check
        if 'val_loss' in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics['val_loss'].item())
