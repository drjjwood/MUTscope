import ast
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm

main_logger = logging.getLogger(__name__)


class SeqInputDataset(Dataset):
    def __init__(self, df, datatype):
        """
        A PyTorch Dataset for loading wild-type or mutant sequences with associated metadata.
        
        This dataset is used to prepare inputs for embedding models, including sequence strings,
        mutation information, and optional hand-crafted feature vectors.
        
        Args:
            df (pandas.DataFrame): A DataFrame containing:
                - '{datatype}_seq': str, the full sequence (wild-type or mutant).
                - '{datatype}_aa': str, the single amino acid at the mutated index.
                - 'target_id': str, sample ID.
                - 'processed_index': int, 0-based index of the mutated residue in the sequence.
                - 'label' (optional): int or float, the sample label.
                - 'feature_vector' (optional): str or list, hand-crafted features (as a stringified list or array).
            datatype (str): Either 'wt' or 'mt' to select the relevant sequence columns.
        
        """
        self.seq = df[f'{datatype}_seq']
        self.aa = df[f'{datatype}_aa']
        self.target_id = df['target_id']
        self.processed_index = df['processed_index']
        if 'label' in df.columns:
            self.label = df['label']
        else:
            self.label = [None] * len(self.seq)

        if 'feature_vector' in df.columns:
            fv_series = df['feature_vector']
            self.feature_vector = [
                ast.literal_eval(x) if isinstance(x, str) else x
                for x in fv_series
            ]
        else:
            self.feature_vector = None
        main_logger.debug(f"Initialized SeqInputDataset with {len(self.seq)} samples (datatype={datatype})")

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        label = self.label.iloc[idx] if hasattr(self.label, "iloc") else self.label[idx]
        seq = self.seq.iloc[idx] if hasattr(self.seq, "iloc") else self.seq[idx]
        aa = self.aa.iloc[idx] if hasattr(self.aa, "iloc") else self.aa[idx]
        target_id = self.target_id.iloc[idx] if hasattr(self.target_id, "iloc") else self.target_id[idx]
        processed_index = self.processed_index.iloc[idx] if hasattr(self.processed_index, "iloc") else self.processed_index[idx]

        feature = None
        if self.feature_vector is not None:
            feature = self.feature_vector.iloc[idx] if hasattr(self.feature_vector, "iloc") else self.feature_vector[idx]
        return (label, seq, aa, target_id, processed_index, feature)


def prepare_input_dataloaders(df, batch_size, num_workers):
    """
    Prepares DataLoaders for wild-type and mutant sequences.
    
    Args:
        df (pandas.DataFrame): Input data containing wild-type and mutant sequences.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of parallel workers for data loading.
    
    Returns:
        tuple: (wt_dataloader, mt_dataloader), each a torch.utils.data.DataLoader.
    """

    wt_dataset = SeqInputDataset(df, datatype="wt")
    mt_dataset = SeqInputDataset(df, datatype="mt")
    wt_dataloader = DataLoader(wt_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers = num_workers)
    mt_dataloader = DataLoader(mt_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers = num_workers)
    main_logger.info(f"Prepared seq_input dataloaders — WT: {len(wt_dataset)} samples, MT: {len(mt_dataset)} samples")
    return wt_dataloader, mt_dataloader

def collate_fn(batch):
    """
    Custom collate function to combine a batch of samples into a single dictionary.
    
    Args:
        batch (list of tuples): Each tuple is (label, sequence, amino_acid, target_id, processed_index, feature_vector).
    
    Returns:
        dict: {
            "labels": List[int or None],
            "sequences": List[str],
            "amino_acids": List[str],
            "target_ids": List[str],
            "processed_indices": List[int],
            "feature_vectors": List[List[float]] or None
        }
    """

    labels, sequences, aa, target_id, processed_index, feature_vector = zip(*batch)
    return {
        "labels": list(labels),
        "sequences": list(sequences),
        "amino_acids": list(aa),
        "target_ids": list(target_id),
        "processed_indices": list(processed_index),
        "feature_vectors": list(feature_vector),
    }


class EncodedOutputDataset:
    """
    Dataset class for handling encoded ESM embeddings and associated metadata.
    
    Args:
        x (torch.Tensor): Embedding tensor of shape (N, D).
        logits (torch.Tensor): Logit tensor of shape (N, 1).
        target_id (List[str]): Sample IDs.
        label (List[int or float], optional): Optional labels for classification tasks.
    """
    def __init__(self, x, logits, target_id, label=None):
        self.x = x
        self.label = label
        self.logits = logits
        self.target_id = target_id

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_tensor = self.x[index]
        logits_tensor = self.logits[index]
        x_with_logits = torch.cat([x_tensor, logits_tensor], dim=0)

        item = {
            'x': x_tensor,
            'logits': logits_tensor,
            'target_id': self.target_id[index],
            'x_with_logits': x_with_logits
        }
        if self.label is not None and self.label[index] is not None:
            try:
                item['label'] = torch.tensor(float(self.label[index]), dtype=torch.float32)
            except (ValueError, TypeError):
                item['label'] = None
        return item

def save_encoded_dataset(dataset, save_path, dataset_name):
    """
    Saves the encoded dataset (embeddings, logits, labels, and IDs) to disk.
    
    Args:
        dataset (EncodedOutputDataset): The dataset object to save.
        save_path (str): Directory to save the file.
        dataset_name (str): Descriptive name used as the filename (e.g., "train").
    """
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{dataset_name}.pt")
    torch.save({
        "x": dataset.x,
        "label": dataset.label,
        "logits": dataset.logits,
        "target_id": dataset.target_id
    }, save_file)
    main_logger.info(f"Saved {dataset_name} dataset to {save_file}")

def load_encoded_dataset(load_path, data_split):
    """
    Loads an EncodedOutputDataset from a .pt file.
    
    Args:
        load_path (str): Path to the saved dataset file (e.g., "train.pt").
        data_split (str): Label for logging purposes (e.g., "train", "val", "test").
    
    Returns:
        EncodedOutputDataset: The loaded dataset with tensors and metadata.
    """
    data = torch.load(load_path)
    main_logger.info(f"Loaded {data_split} dataset from {load_path}")
    return EncodedOutputDataset(
        x=data["x"],
        label=data["label"],
        logits=data["logits"],
        target_id=data["target_id"]
    )
