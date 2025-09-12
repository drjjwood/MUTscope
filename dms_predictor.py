import os
import pandas as pd

AA_LIST = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def load_fasta(file_path):
    """
    Load a FASTA file and return the FASTA ID and sequence.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip()
    if not header.startswith('>'):
        raise ValueError("The FASTA file must start with a '>' character.")
    fasta_id = header[1:].strip()
    
    sequence = ''.join(line.strip().strip("'") for line in lines[1:])
    
    return fasta_id, sequence

def generate_substitutions(wt_sequence, possible_values):
    """
    Generate every single substitution for the input sequence.
    
    Args:
        wt_sequence (str): Original sequence.
        possible_values (list): List of possible characters for substitution.
        
    Returns:
        List of tuples: Each tuple contains (mt_sequence, protein_change).
    """
    substitutions = []
    seq_list = list(wt_sequence)
    
    for idx, original_letter in enumerate(seq_list):
        for new_letter in possible_values:
#            if new_letter == original_letter:
#                continue
            mutated = seq_list.copy()
            mutated[idx] = new_letter
            mt_sequence = "".join(mutated)
            protein_change = f"{original_letter}{idx+1}{new_letter}"
            substitutions.append((mt_sequence, protein_change))
    
    return substitutions

def create_dataframe(fasta_id, wt_sequence, substitutions):
    """
    Create a DataFrame with each substitution.
    
    Args:
        fasta_id (str): The FASTA ID.
        wt_sequence (str): Wild-type sequence.
        substitutions (list): List of tuples (mt_sequence, protein_change).
        
    Returns:
        pandas.DataFrame: DataFrame with columns for the FASTA id, wild-type sequence,
                          mutated sequence, and derived protein change details.
    """
    df = pd.DataFrame(substitutions, columns=['mt_seq', 'protein_change'])
    df['wt_seq'] = wt_sequence

    df['target_id'] = fasta_id
    df['target_id'] = df['target_id'] + '.' + df['protein_change']
    
    df['wt_aa'] = df['protein_change'].str[0]
    df['aa_index'] = df['protein_change'].str[1:-1]
    df['mt_aa'] = df['protein_change'].str[-1]
    
    df = df.drop(columns='protein_change')
    return df


def create_dms_df(predict_config):
    in_path = os.path.join(predict_config.dataset.input_path, f"{predict_config.predictions.fasta_file_name}.fasta")
    out_path = os.path.join(predict_config.dataset.input_path, f"{predict_config.dataset.predict_file_name}.csv")
    
    if os.path.exists(out_path):
        raise FileExistsError(f"Aborting : Output file {out_path} already exists. Please use different path or remove file")
    
    fasta_id, wt_sequence = load_fasta(in_path)
    print(f"FASTA ID: {fasta_id}")
    print(f"Wild-type sequence: {wt_sequence}")
    
    substitutions = generate_substitutions(wt_sequence, AA_LIST)
    df = create_dataframe(fasta_id, wt_sequence, substitutions)
    
    df.to_csv(out_path, index=False)
    print(f"DataFrame saved to {out_path}")

