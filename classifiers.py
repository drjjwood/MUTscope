import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import wandb

def initialize_classifier(classifier_type, embedding_size, config):
    """
    Factory function to initialize a classifier based on the specified type.

    Args:
        classifier_type (str): Type of classifier ('CNN', 'MLP1lyr', or 'MLP2lyr').
        embedding_size (int): Size of the input embedding vector.
        config (Namespace): Configuration object containing hyperparameters.

    Returns:
        nn.Module: Instantiated classifier model.
    """
    if classifier_type =='CNN':
        return initialize_classifier_CNN(embedding_size, config)
    elif classifier_type =='MLP1lyr':
        return initialize_classifier_MLP(embedding_size, config, nlyr=1)
    elif classifier_type =='MLP2lyr':
        return initialize_classifier_MLP(embedding_size, config, nlyr=2)
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")

def initialize_classifier_MLP(embedding_size, config, nlyr=1):
    """
    Initializes a multilayer perceptron (MLP) classifier with one or two layers.

    Args:
        embedding_size (int): Input feature dimension.
        config: Configuration object containing settings, including sweep info.
        nlyr (int): Number of layers (1 or 2).

    Returns:
        nn.Module: Initialized MLP classifier.

    Raises:
        ValueError: If nlyr is not 1 or 2.
    """
    if config.settings.run_sweep:
        sweep_config = wandb.config
        num_hidden1 = sweep_config.num_hidden1
        num_hidden2 = sweep_config.num_hidden2 if nlyr == 2 else None
        dropout = sweep_config.dropout
    else:
        num_hidden1 = int(embedding_size / 2)
        dropout = 0.5

    if nlyr == 1:
        return MLPClassifier_LeakyReLu(
            num_input=embedding_size,
            num_hidden1=num_hidden1,
            dropout=dropout,
            num_output=1,
        )
    elif nlyr == 2:
        num_hidden2 = (int(embedding_size / 2)) // 2
        return MLPClassifier_2lyr_LeakyReLu(
            num_input=embedding_size,
            num_hidden1=num_hidden1,
            num_hidden2=num_hidden2,
            dropout=dropout,
            num_output=1,
        )
    else:
        raise ValueError("nlyr must be either 1 or 2.")


class MLPClassifier_LeakyReLu(nn.Module):
    def __init__(self, num_input, num_hidden1, dropout, num_output):
        """
        Initializes a single-layer MLP classifier with LeakyReLU activation.

        Args:
            num_input (int): Dimensionality of the input vector.
            num_hidden1 (int): Number of hidden units in the first layer.
            dropout (float): Dropout probability applied after the first layer.
            num_output (int): Output dimensionality (usually 1 for binary classification).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input, num_hidden1),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden1, num_output)
        )
    
    def forward(self, x):
        return self.model(x)

class MLPClassifier_2lyr_LeakyReLu(nn.Module):
    def __init__(self, num_input, num_hidden1, num_hidden2, dropout, num_output):
        """
        Initializes a two-layer MLP classifier with batch normalization and LeakyReLU activations.

        Args:
            num_input (int): Dimensionality of the input vector.
            num_hidden1 (int): Number of hidden units in the first hidden layer.
            num_hidden2 (int): Number of hidden units in the second hidden layer.
            dropout (float): Dropout probability applied after the second hidden layer.
            num_output (int): Output dimensionality (typically 1 for binary classification).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input, num_hidden1),
            nn.BatchNorm1d(num_hidden1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden1, num_hidden2),
            nn.BatchNorm1d(num_hidden2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_hidden2, num_output)
        )
    
    def forward(self, x):
        return self.model(x)

def initialize_classifier_CNN(embedding_size, config):
    """
    Create and return a CNN classifier instance.
    
    Args:
        embedding_size (int): Length of the input embedding (1D vector).
        config: Configuration object that may contain hyperparameters.
    
    Returns:
        An instance of CNNClassifier.
    """
    if config.settings.run_sweep:
        sweep_config = wandb.config
        num_filters1 = sweep_config.num_filters1
        num_filters2 = sweep_config.num_filters2
        kernel_size1 = sweep_config.kernel_size1
        kernel_size2 = sweep_config.kernel_size2
        dropout = sweep_config.dropout
    else:
        num_filters1 = max(16, embedding_size // 8)
        num_filters2 = max(16, num_filters1 // 2)
        kernel_size1 = 5
        kernel_size2 = 3
        dropout = 0.5

    return CNNClassifier(
        num_input=embedding_size,
        num_filters1=num_filters1,
        num_filters2=num_filters2,
        kernel_size1=kernel_size1,
        kernel_size2=kernel_size2,
        dropout=dropout,
        num_output=1,
    )

class CNNClassifier(nn.Module):
    def __init__(self, num_input, num_filters1, num_filters2, kernel_size1, kernel_size2, dropout, num_output):
        """
        Args:
            num_input (int): Length of the input vector.
            num_filters1 (int): Number of filters in the first convolutional layer.
            num_filters2 (int): Number of filters in the second convolutional layer.
            kernel_size1 (int): Kernel size for the first conv layer.
            kernel_size2 (int): Kernel size for the second conv layer.
            dropout (float): Dropout probability.
            num_output (int): Number of output units (1 for binary classification).
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters1, kernel_size=kernel_size1)
        self.bn1 = nn.BatchNorm1d(num_filters1)
        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size2)
        self.bn2 = nn.BatchNorm1d(num_filters2)
        
        L1 = num_input - kernel_size1 + 1
        L2 = L1 - kernel_size2 + 1
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters2 * L2, num_output)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
