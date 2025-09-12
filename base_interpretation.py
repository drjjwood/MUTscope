import numpy as np
import pandas as pd
import seaborn as sns
from plots_interpretation import category_histplot, plot_histogram_with_categories

sns.set_theme(style="white")

def prepare_preds_df(preds):
    """
    Parse the target_id field into organism, gene, mutation, wt/mt amino acids, and index.

    Args:
        preds (pd.DataFrame): Predictions dataframe with column 'target_id' (format: org.gene.mutation).

    Returns:
        pd.DataFrame: DataFrame with additional parsed columns.
    """
    preds['organism'] = preds['target_id'].apply(lambda x: x.split('.')[0])
    preds['gene'] = preds['target_id'].apply(lambda x: x.split('.')[1])
    preds['protein_change'] = preds['target_id'].apply(lambda x: x.split('.')[2])
    preds['wt_aa'] = preds['protein_change'].str[0]
    preds['mt_aa'] = preds['protein_change'].str[-1]
    preds['aa_index'] = preds['protein_change'].str[1:-1].astype(int)
    return preds

def run_basic_preds_analysis(preds_df, analysis_type='gene', save_path = None):
    """
    Run basic prediction interpretation including category assignment and histogram plotting.

    Args:
        preds_df (pd.DataFrame): Prediction DataFrame including probabilities and labels.
        analysis_type (str): Grouping level for summary (e.g., 'gene', 'organism').
        save_path (str, optional): Path to save outputs. If None, nothing is saved.

    Returns:
        Tuple of:
            - pd.DataFrame: Enriched prediction DataFrame.
            - pd.DataFrame: Summary pivot table by group.
    """
    preds_df = assign_classification_group(preds_df, 'binary_predictions')
    preds_summary = get_preds_summary_table(preds_df, analysis_type)
    category_histplot(preds_df, analysis_type, "category", ['TN','TP','FN','FP'], save_path=save_path)
    plot_histogram_with_categories(preds_df, 'probabilities', save_path=save_path)#, common_norm=False)
    if save_path:
        preds_df.sort_values(by='probabilities', ascending=False).to_csv(f'{save_path}/preds_df.csv', index=False)
        preds_summary.to_csv(f'{save_path}/preds_summary.csv')
    return preds_df, preds_summary

def assign_classification_group(df, prediction_col='prediction'):
    """
    Assign TP/FP/TN/FN classification labels based on true and predicted labels.

    Args:
        df (pd.DataFrame): DataFrame containing 'labels' and prediction column.
        prediction_col (str): Name of the column containing binary predictions.

    Returns:
        pd.DataFrame: DataFrame with a new 'category' column.
    """
    conditions = [
        (df['labels'] == 1) & (df[prediction_col] == 0),
        (df['labels'] == 0) & (df[prediction_col] == 1),
        (df['labels'] == 1) & (df[prediction_col] == 1),
        (df['labels'] == 0) & (df[prediction_col] == 0)
    ]
    choices = ["FN", "FP", "TP", "TN"]
    df["category"] = np.select(conditions, choices, default="ERROR")
    
    # Ensure that the category column includes all classification groups, even if absent in the data.
    fixed_categories = ["FN", "FP", "TP", "TN"]
    df["category"] = pd.Categorical(df["category"], categories=fixed_categories, ordered=True)
    
    return df

def get_preds_summary_table(preds, analysis_type):
    """
    Generate a pivot table summarizing prediction outcomes (TP/TN/FP/FN) by group.

    Args:
        preds (pd.DataFrame): Enriched DataFrame with classification categories.
        analysis_type (str): Grouping level for summary (e.g., 'gene', 'organism').

    Returns:
        pd.DataFrame: Summary pivot table including counts, proportions, and accuracy.
    """
    preds_count = preds[['category', analysis_type]].value_counts().reset_index(name='count').sort_values(by='category')
    preds_pivot = preds_count.pivot_table(values='count',
                                          index=analysis_type, 
                                          columns='category',aggfunc="sum").sort_values(by='TP', ascending=False)
    preds_pivot.replace(np.nan,0, inplace=True)
    preds_pivot['n_pos'] = preds_pivot['TP']+preds_pivot['FN']
    preds_pivot['n_neg'] = preds_pivot['TN']+preds_pivot['FP']
    preds_pivot['proportion_pos'] = preds_pivot['n_pos']/preds_pivot['n_neg']
    preds_pivot['total'] = (preds_pivot['n_pos']+preds_pivot['n_neg'])
    preds_pivot['acc'] = np.round((preds_pivot['TP']+preds_pivot['TN'])/preds_pivot['total']*100,2)
    preds_pivot.sort_values(by= 'acc', ascending=False)
    return preds_pivot