import os
import wandb
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from evaluation import extract_data
from evaluation_classical import evaluate_classical_dataset
from imblearn.ensemble import BalancedRandomForestClassifier
from training_utils import setup_logger, get_model_save_path
main_logger = logging.getLogger(__name__)

CLASSICAL_MODELS = {
    "LR": lambda: LogisticRegression(C=1.0, max_iter=1000, penalty='l2', solver='lbfgs'),
    "SVC": lambda: LinearSVC(C=1.0, max_iter=1000),
    "GBC": lambda: GradientBoostingClassifier(max_depth=6, n_estimators=100, learning_rate=0.1),
    "BRF": lambda: BalancedRandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=20,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=42)}

def train_classical_model(train_loader, classifier_type, config):
    """
    Train a classical model on the entire training set.

    Args:
        train_loader: DataLoader for training data.
        classifier_type (str): One of 'LR', 'SVC', 'GBC' or 'BRF'
        config: Configuration object containing model settings.

    Returns:
        model: A trained classical model
    """
    _, _, train_features, train_labels, _ = extract_data(train_loader, outputs=None, include_labels=True)
    X = train_features
    y = train_labels
    main_logger.info("Training classical model with features shape:", X.shape)
    try:
        model = CLASSICAL_MODELS[classifier_type]()
        main_logger.info(f"Using classifier: {classifier_type}")
    except KeyError:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    main_logger.info("Fitting classical model...")
    model.fit(X, y)
    
    return model

def save_classical_model(model, save_path):
    """
    Save the trained classical model to disk using pickle.

    Args:
        model: The trained classical model.
        save_path (str): File path to save the model.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    main_logger.info(f"Model saved to {save_path}")

def train_classical_pipeline(train_loader, val_loader, test_loader, config):
    """
    Run the complete training pipeline for classical models.

    This function:
      1) Trains the model
      2) Logs to WandB via your logger
      3) Saves the trained model under
         <config.model.save_path>/<experiment_name>/<classifier>_model.pkl
      4) Evaluates on train/val/test using that saved file

    Returns:
        (model, model_save_path)
    """
    logger = setup_logger(config)
    classifier_type = config.model.classifier_head
    model = train_classical_model(train_loader, classifier_type, config)

    model_save_path = get_model_save_path(config, classifier_type)
    save_classical_model(model, model_save_path)
    
    main_logger.info(f"Evaluating {classifier_type} model...")
    evaluate_classical_dataset(
        train_loader,
        val_loader,
        test_loader,
        config,
        model_save_path,
        logger
    )

    return model, model_save_path
