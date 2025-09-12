import pandas as pd
from base_interpretation import run_basic_preds_analysis, prepare_preds_df
from dimensionality_interpretation import dim_rdxn_analysis
from plots_interpretation import plot_hist_probabilities
import os
import logging
main_logger = logging.getLogger(__name__)

def run_training_interpretation(config):
    """
    Run interpretation for all labeled datasets: train, validation, and test.

    Automatically loads predictions for each split and saves interpretability outputs 
    (e.g., PCA plots, summary metrics) under the experiment directory.

    Args:
        config: Configuration object with dataset and experiment paths.
    """
    for dataset_name in (
        config.dataset.train_file_name,
        config.dataset.val_file_name,
        config.dataset.test_file_name
    ):
        pred_filepath, save_path = get_prediction_paths(config, dataset_name)
        main_logger.info(f"Running interpretation with labels for{dataset_name}...")
        run_label_interpretation(pred_filepath, save_path)


def run_prediction_interpretation(predict_config):
    """
    Run interpretation for a standalone prediction set.

    Automatically detects whether ground-truth labels are present in the prediction file
    and runs the appropriate label-aware or label-free interpretation routine.

    Args:
        predict_config: Configuration object for the prediction run.
    """   
    pred_filepath, save_path = get_prediction_paths(predict_config, dataset_name=predict_config.dataset.predict_file_name)
    preds_df = pd.read_csv(pred_filepath)
    if 'labels' in preds_df.columns:
        main_logger.info(f"Running inference with labels...")
        run_label_interpretation(pred_filepath, save_path)
    else:
        main_logger.info(f"Running inference without labels...")
        run_nolabel_interpretation(preds_df, save_path)

def get_prediction_paths(config, dataset_name):
    """
    Construct prediction file and save directory paths for a given dataset.

    Args:
        config: Configuration object containing dataset and experiment paths.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test', or custom prediction file name).

    Returns:
        Tuple[str, str]: 
            - pred_filepath: Full path to the prediction CSV file.
            - save_path: Directory path to save interpretation outputs.
    """
    preds_base = os.path.join(config.dataset.save_path, config.settings.experiment_name, "predictions")
    pred_filepath = os.path.join(preds_base, f"{dataset_name}_predict.csv")
    save_path = os.path.join(preds_base, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    return pred_filepath, save_path

def run_label_interpretation(pred_filepath, save_path):
    """
    Perform interpretation analysis on predictions with ground-truth labels.

    Runs classification metrics, PCA analysis, and saves visual summaries.

    Args:
        pred_filepath (str): Path to the prediction CSV file (must include 'labels' column).
        save_path (str): Directory where interpretation outputs will be saved.
    """
    preds_df = pd.read_csv(pred_filepath)
    preds_df = prepare_preds_df(preds_df)
    preds_df, preds_summary = run_basic_preds_analysis(preds_df, analysis_type='gene', save_path=save_path)
    dim_rdxn_analysis(preds_df, categories = ['labels','category','organism','gene','wt_aa','mt_aa'], save_path=save_path)
   
def run_nolabel_interpretation(preds_df, save_path=None):
    """
    Perform unsupervised interpretation on a prediction set without ground-truth labels.

    Includes probability histogram and PCA-based embedding visualization.

    Args:
        preds_df (pd.DataFrame): DataFrame containing prediction outputs.
        save_path (str, optional): Directory to save visualizations and processed prediction file. Defaults to None.
    """
    preds_df = prepare_preds_df(preds_df)
    if save_path:
        preds_df.to_csv(f'{save_path}/preds_df.csv', index=False)
    plot_hist_probabilities(preds_df, save_path)
    dim_rdxn_analysis(preds_df, categories = ['organism','gene','wt_aa','mt_aa'], save_path=save_path)
   


        
    

