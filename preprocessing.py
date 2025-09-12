import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from data_io import create_data_paths, load_from_csv

main_logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 1020 # make it an even number
WINDOW_SIZE = MAX_SEQ_LENGTH // 2

def prepare_input_dfs(config):
    """
    Load CSV files for train, validation, and test datasets, format their columns, and optionally randomize a control column.

    Args:
        config: Configuration object containing dataset paths, file names, and settings.

    Returns:
        tuple: (train_df, val_df, test_df) processed DataFrames.
    """
    train_path, val_path, test_path = create_data_paths(config, '.csv')
    train_df_raw, val_df_raw, test_df_raw = load_from_csv(train_path, val_path, test_path)

    train_df = format_df_cols(train_df_raw, config.settings.embedding_type)
    val_df = format_df_cols(val_df_raw, config.settings.embedding_type)
    test_df = format_df_cols(test_df_raw, config.settings.embedding_type)

    # if config.settings.randomised_control_run:
    #     train_df = randomise_column(train_df)

    main_logger.info(f"Train rows after perpare df: {len(train_df_raw)}")
    main_logger.info(f"Val rows after perpare df: {len(val_df_raw)}")
    main_logger.info(f"Test rows after perpare df: {len(test_df_raw)}")
    
    return train_df, val_df, test_df

def create_paired_data(df)-> pd.DataFrame:
    """Duplicate positive-class rows with neutral mutations for negative class pairing."""
    pos_data = df[df['label']==1]
    main_logger.info(f'positive_df={pos_data.shape}')
    df_copy = pos_data.copy()
    df_copy['mut_sequence'] = df_copy['wt_sequence']
    df_copy['mutation'] = df_copy['wt']
    df_copy['label'] = 0
    df_copy['antibiotic_codes'] = np.nan
    df_copy['antibiotic_standardised'] = np.nan
    df_copy['protein_change'] = df_copy['wt']+df_copy['position'].astype(int).astype(str)+df_copy['mutation']
    df_copy['target_id'] = df_copy['organism'] +'.'+ df_copy['gene']+ '.' + df_copy['protein_change']
    paired_df = pd.concat([df,df_copy])
    main_logger.info(f'paired_df={paired_df.shape}')
    return paired_df

def format_df_cols(df, embed_type)-> pd.DataFrame:
    """Rename columns for amr_drone data based on embed_type."""
    df = df.rename(columns={
            'wt': 'wt_aa',
            'mutation': 'mt_aa',
            'wt_sequence': 'wt_seq',
            'position': 'aa_index',
            'mut_sequence': 'mt_seq'
        })
    desired_cols = ['target_id', 'wt_aa', 'mt_aa', 'label','wt_seq', 'aa_index', 'mt_seq', 'feature_vector']
    cols = [c for c in desired_cols if c in df.columns]
    df = df[cols]
    return df

def randomise_column(df: pd.DataFrame, col: str = 'label') -> pd.DataFrame:
    """
    Randomizes the values in the specified column of the DataFrame.
    """
    df[col] = np.random.permutation(df[col].values)
    return df

def seq_length_process(df)-> pd.DataFrame:
    """
    Process sequences to ensure they are no longer than MAX_SEQ_LENGTH.
    Handles truncation and updates the processed index.
    Args:
        df: DataFrame containing sequences and related metadata.
    Returns:
        Processed DataFrame with truncated sequences and updated indices.
    """
    main_logger.info(f"Data length before length processing: {df.shape[0]}")
    
    df = df.dropna(subset=["wt_seq", "mt_seq", "aa_index"]).copy()
    df['aa_index'] = df['aa_index'].astype(int)
    df['Length'] = df['wt_seq'].str.len()

    if df['Length'].max() > MAX_SEQ_LENGTH:
        remain_df = df[df['Length'] <= MAX_SEQ_LENGTH]
        remain_df['processed_index'] = remain_df['aa_index']
        remain_df['case'] = 0 # For debugging if needed
        
        trunc_df = df[df['Length'] > MAX_SEQ_LENGTH]
        truncated_df = get_truncation(trunc_df)
        df = pd.concat([remain_df, truncated_df]).reset_index(drop=True)
    else:
        df['processed_index'] = df['aa_index']
    
    df['Truncated_length'] = df['wt_seq'].str.len()
    main_logger.info(f"Data length after length processing: {df.shape[0]}")
    return df

def get_truncation(df)-> pd.DataFrame:
    """
    Truncate long sequences to a maximum length while preserving the context
    around a specific amino acid index (aa_index).
    Args:
        df: DataFrame containing sequences to truncate.
    Returns:
        DataFrame with truncated sequences and updated indices.
    """
    main_logger.info(f"The number of sequences to be truncated: {len(df)}")

    has_label = 'label' in df.columns
    has_fv = 'feature_vector' in df.columns

    truncated_sequences = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq_len   = row['Length']
        aa_index  = row['aa_index']

        # Case 1: near start
        if aa_index < WINDOW_SIZE:
            wt_seq = row['wt_seq'][:MAX_SEQ_LENGTH]
            mt_seq = row['mt_seq'][:MAX_SEQ_LENGTH]
            processed_index = aa_index
            case = 1

        # Case 2: near end
        elif aa_index >= seq_len - WINDOW_SIZE:
            wt_seq = row['wt_seq'][-MAX_SEQ_LENGTH:]
            mt_seq = row['mt_seq'][-MAX_SEQ_LENGTH:]
            processed_index = aa_index - seq_len + MAX_SEQ_LENGTH
            case = 2

        # Case 3: middle
        else:
            start = max(0, aa_index - WINDOW_SIZE)
            if start % 2 != 0 and start > 0:
                start -= 1
            end = start + MAX_SEQ_LENGTH
            if end > seq_len:
                start = max(0, seq_len - MAX_SEQ_LENGTH)
                end = seq_len
            wt_seq = row['wt_seq'][start:end]
            mt_seq = row['mt_seq'][start:end]
            processed_index = aa_index - start
            case = 3
        assert wt_seq[processed_index - 1] == row['wt_aa'], \
        f"Mismatch: expected {row['wt_aa']} at processed_index={processed_index}, found {wt_seq[processed_index - 1]}"

        # build the base entry
        entry = {
            'target_id':       row['target_id'],
            'aa_index':        row['aa_index'],
            'processed_index': int(processed_index),
            'wt_aa':           row['wt_aa'],
            'mt_aa':           row['mt_aa'],
            'wt_seq':          wt_seq,
            'mt_seq':          mt_seq,
            'Length':          seq_len,
            'case':            case,
        }

        # only if the original df had it do we propagate it
        if has_label:
            entry['label'] = row['label']
        if has_fv:
            entry['feature_vector'] = row['feature_vector']

        truncated_sequences.append(entry)

    return pd.DataFrame(truncated_sequences)

