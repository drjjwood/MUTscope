import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
import logging
from sklearn.decomposition import PCA
from tqdm import tqdm
import pytorch_lightning as pl  
from esm import Alphabet, pretrained
import torch.nn as nn
import pickle
from base_embeddings import prepare_input_dataloaders, EncodedOutputDataset, save_encoded_dataset

main_logger = logging.getLogger(__name__)

def load_esm_model(config, device):
    """
    Loads an ESM model and its associated alphabet based on the specified embedding type in the config.
    
    Args:
        config: Configuration object with settings.embedding_type attribute ('esm2' or other supported types).
        device (torch.device): Device to load the model on ('cuda' or 'cpu').
    
    Returns:
        tuple: (esm_model, alphabet) where esm_model is a torch.nn.Module and alphabet is the associated ESM Alphabet.
    """

    if config.settings.embedding_type == 'esm2':
        esm_model, alphabet = pretrained.esm2_t36_3B_UR50D()
    else:
        esm_model, alphabet = pretrained.esm1b_t33_650M_UR50S()
    esm_model = esm_model.to(device)
    esm_model.eval()
    return esm_model, alphabet

def generate_embeds_and_save(df, save_path, dataset_filename, model, tokenizer, device, config):
    """
    Generates embeddings for the sequences in the given DataFrame using the provided model and tokenizer, 
    saves the encoded dataset to disk, and returns the EncodedOutputDataset.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the sequences and associated metadata.
        save_path (str): Directory path where the encoded dataset will be saved.
        dataset_filename (str): Name identifier for the dataset (e.g., "train", "val", "test").
        model (torch.nn.Module): The model used to generate embeddings.
        tokenizer: The tokenizer or alphabet required by the model (e.g., from ESM).
        device (torch.device): Device on which to run the model (e.g., "cuda" or "cpu").
        config (dict or EasyDict): Configuration object containing dataset parameters such as batch size and number of workers.

    Returns:
        EncodedOutputDataset: An object containing the generated embeddings, logits, labels, and target IDs.
    """
    model = model.to(device)
    main_logger.info(f"Preparing {dataset_filename} embedding dataloaders...")
    wt_dataloader, mt_dataloader = prepare_input_dataloaders(df, batch_size=config.dataset.loader_batch_size, num_workers = config.dataset.num_workers)

    final_result = generate_embeddings_data(
        wt_dataloader, mt_dataloader, model, tokenizer, device, config
    )

    if config.pca.dim_reduction==True:
        final_result = run_pca_reduction(final_result, dataset_filename, config)

    encoded_dataset = EncodedOutputDataset(
        x=final_result['x'], 
        logits=final_result['logits'],
        target_id=final_result['target_id'], 
        label=final_result['label'])

    save_encoded_dataset(encoded_dataset, save_path, dataset_filename)
    return encoded_dataset


def run_pca_reduction(final_result, dataset_filename, config):
    """
    Performs PCA dimensionality reduction on embedding tensors. Saves or loads the PCA model depending on dataset type.
    
    Args:
        final_result (dict): Dictionary with key 'x' containing embedding tensor.
        dataset_filename (str): Used to determine whether PCA model should be trained ('train') or loaded ('val'/'test').
        config: Configuration object with PCA and save path settings.
    
    Returns:
        dict: Updated final_result dictionary with reduced 'x' tensor.
    """

    x_np = final_result['x'].numpy()
    pca_model_filename = f"pca_model_{config.settings.experiment_name}.pkl"
    pca_model_filepath = os.path.join(config.dataset.save_path, config.settings.experiment_name, pca_model_filename)

    if 'train' in dataset_filename:
        pca = PCA(n_components=config.pca.n_components, random_state=config.settings.seed)
        x_reduced = pca.fit_transform(x_np)
        with open(pca_model_filepath, 'wb') as f:
            pickle.dump(pca, f)
    else:
       # For validation/test sets, load the PCA model fitted on train set from disk
        with open(pca_model_filepath, 'rb') as f:
           pca = pickle.load(f)
        x_reduced = pca.transform(x_np)

    final_result['x'] = torch.tensor(x_reduced, dtype=torch.float32)
    return final_result
    

def generate_embeddings_data(
    wt_dataloader: DataLoader,
    mt_dataloader: DataLoader,
    model: nn.Module,
    alphabet,
    device: torch.device,
    config,
    log_progress: bool = True,
) -> dict:
    """
    Generates ESM embeddings and computed logits for batches of wild-type and mutant sequences.
    
    Args:
        wt_dataloader (DataLoader): DataLoader for wild-type sequences.
        mt_dataloader (DataLoader): DataLoader for mutant sequences.
        model (torch.nn.Module): ESM model to generate embeddings.
        alphabet: ESM Alphabet object with batch_converter and vocab info.
        device (torch.device): Device to run inference on.
        config: Configuration object for embedding settings.
        log_progress (bool): If True, shows progress bar with tqdm.
    
    Returns:
        dict: {
            "x": torch.Tensor, concatenated embeddings (N, D),
            "logits": torch.Tensor, mutation logits (N, 1),
            "label": List[int or float],
            "target_id": List[str]
        }
    """

    batch_converter = alphabet.get_batch_converter()
    esm_dict = alphabet.tok_to_idx

    labels, target_ids, embeddings, logits, features = [], [], [], [], []

    dataloader_iter = zip(wt_dataloader, mt_dataloader)
    if log_progress:
        dataloader_iter = tqdm(dataloader_iter, total=len(wt_dataloader), desc="Generating embeddings")

    for batch_idx, (wt_batch, mt_batch) in enumerate(dataloader_iter):
        wt_tokenized = get_tokenised_batch_data(wt_batch, batch_converter, device, 'wt')
        mt_tokenized = get_tokenised_batch_data(mt_batch, batch_converter, device, 'mt')

        if batch_idx == 0: # To track a singe batch for transparancy
            main_logger.debug(f"Tokenized WT batch example: {wt_tokenized['tokens'][0][:10]}... (truncated)")
            main_logger.debug(f"Processed index: {wt_tokenized['processed_index'][0]}")
        
        wt_embeds, mt_embeds, batch_logits = get_positional_embeddings(
            model=model,
            wt_tokens=wt_tokenized["tokens"],
            mt_tokens=mt_tokenized["tokens"],
            wt_metadata=wt_tokenized,
            mt_metadata=mt_tokenized,
            esm_dict=esm_dict,
            device=device,
            config=config
        )

        if batch_idx == 0: # To track a singe batch for transparancy:
            main_logger.debug(f"WT embedding shape: {wt_embeds.shape}")
            main_logger.debug(f"MT embedding shape: {mt_embeds.shape}")
            main_logger.debug(f"Batch logits shape: {batch_logits.shape}")
        
        embeddings.append(torch.cat((wt_embeds, mt_embeds), dim=1))
        logits.append(batch_logits)
        labels.extend(wt_tokenized["labels"])
        target_ids.extend(wt_tokenized["target_id"])
        
        if "feature_vector" in wt_tokenized and wt_tokenized["feature_vector"] is not None:
            features.append(torch.tensor(wt_tokenized["feature_vector"], dtype=torch.float32).to(device))

    final_x = torch.cat(embeddings, dim=0).cpu().detach()
    final_logits = torch.cat(logits, dim=0).cpu().detach()

    # Concatenate with hand-curated features if present
    if features:
        feature_tensor = torch.cat(features, dim=0).cpu().detach()
        final_x = torch.cat((final_x, feature_tensor), dim=1)

    main_logger.info(f'final X shape {final_x.shape}')
    main_logger.info(f'final logits shape {final_logits.shape}')

    return {
        "x": final_x,
        "logits": final_logits,
        "label": labels,
        "target_id": [str(x) for x in target_ids],
    }


def get_tokenised_batch_data(batch, batch_converter, device, datatype):
    """
    Tokenizes a batch of sequences using the ESM batch_converter and prepares metadata.
    
    Args:
        batch (dict): A batch from the DataLoader containing labels, sequences, indices, etc.
        batch_converter: Function to convert (label, sequence) pairs into ESM-compatible token tensors.
        device (torch.device): Device to move tensors to.
        datatype (str): Indicates whether the data is "wt" or "mt" (for logging or debugging).
    
    Returns:
        dict: {
            "labels": list of labels,
            "tokens": torch.Tensor of shape (batch_size, seq_len),
            "aa": list of amino acids at mutated position,
            "target_id": list of sequence IDs,
            "processed_index": list of integer indices,
            "feature_vector": torch.Tensor or None
        }
    """
    _, _, tokens = batch_converter(list(zip(batch["labels"], batch["sequences"])))

    raw_fv = batch.get("feature_vectors", None)
    if raw_fv is None or any(elem is None for elem in raw_fv):
        fv_tensor = None
    else:
        fv_tensor = torch.tensor(raw_fv, dtype=torch.float32).to(device)

    return {
        "labels":           batch["labels"],
        "tokens":           tokens,
        "aa":               batch["amino_acids"],
        "target_id":        batch["target_ids"],
        "processed_index":  batch["processed_indices"],
        "feature_vector":   fv_tensor,
    }


def get_positional_embeddings(
    model: nn.Module,
    wt_tokens: torch.Tensor,
    mt_tokens: torch.Tensor,
    wt_metadata: dict,
    mt_metadata: dict,
    esm_dict: dict[str, int],
    device: torch.device,
    config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates positional embeddings for wild-type and mutant sequences, and computes logits based on the extracted representations.

    Args:
        model (torch.nn.Module): Model used to generate embeddings.
        wt_tokens (torch.Tensor): Tokenized tensor for wild-type sequences.
        mt_tokens (torch.Tensor): Tokenized tensor for mutant sequences.
        wt_metadata (dict): Metadata for wild-type data including processed indices and additional features.
        mt_metadata (dict): Metadata for mutant data including processed indices and additional features.
        esm_dict (dict[str, int]): Dictionary mapping ESM tokens to indices.
        device (torch.device): Device to run the model on.
        last_layer (int): Layer number from which to extract representations (default is 33).

    Returns:
        tuple: A tuple containing:
            - wt_repr (torch.Tensor): Wild-type representations at the processed indices.
            - mt_repr (torch.Tensor): Mutant representations at the processed indices.
            - logits (torch.Tensor): Logits computed as the log ratio of mutant to wild-type logits.
    """
    layer_dict = {'esm':33, 'esm1b':33, 'esm2':36}
    last_layer = layer_dict[config.settings.embedding_type]

    processed_index = torch.tensor(wt_metadata['processed_index']).to(device)
    batch_indices = torch.arange(len(processed_index))

    with torch.no_grad():
        # get full embeds
        wt_result = model(wt_tokens.to(device), repr_layers=[last_layer])
        mt_result = model(mt_tokens.to(device), repr_layers=[last_layer])
        
        # Slice representations to include only relevant positions
        wt_repr = wt_result["representations"][last_layer][batch_indices, processed_index]
        mt_repr = mt_result["representations"][last_layer][batch_indices, processed_index]

        # Slice logits to include only relevant positions
        wt_logits_raw = wt_result["logits"][batch_indices, processed_index]
        mt_logits_raw = mt_result["logits"][batch_indices, processed_index]

        # Extract logits for amino acids of interest
        wt_logits = extract_logits(wt_logits_raw, wt_metadata["aa"], esm_dict, batch_indices)
        mt_logits = extract_logits(mt_logits_raw, mt_metadata["aa"], esm_dict, batch_indices)
        
        # Log ratio of ESM-predicted probabilities (mutant vs wild-type)
        logits_ratio = torch.log((mt_logits + 1e-8) / (wt_logits + 1e-8)).unsqueeze(1)

    return wt_repr, mt_repr, logits_ratio

def extract_logits(
    total_logits: torch.Tensor,
    amino_acids: list[str],
    esm_dict: dict[str, int],
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts logits for specific amino acids from a tensor of raw logits using a provided ESM dictionary.

    This function applies a softmax to the raw logits and then selects the logits corresponding 
    to amino acids that are found in the provided `esm_dict`. The selection is based on the indices 
    of the amino acids in the input list, aligned with the batch indices.

    Args:
        total_logits (torch.Tensor): The raw logits tensor with shape (batch_size, vocab_size)
            containing the unnormalized log probabilities.
        amino_acids (list[str]): A list of amino acid characters (one per position in the sequence) 
            for the batch.
        esm_dict (dict[str, int]): A dictionary mapping amino acid characters to their corresponding indices 
            in the vocabulary.
        batch_indices (torch.Tensor): A tensor containing the indices for the batch.

    Returns:
        torch.Tensor: A tensor containing the extracted logits for the specified amino acids. If no valid 
        entries are found (i.e., none of the amino acids are present in `esm_dict`), returns an empty tensor.
    """
    softmax = nn.Softmax(dim=-1)
    # Identify, for each sample, which amino acid maps to which index in the vocabulary, if no amino acid is valid, return an empty tensor.
    valid_entries = [(i, esm_dict[aa]) for i, aa in enumerate(amino_acids) if aa in esm_dict]
    if not valid_entries:
        return torch.empty(0, dtype=total_logits.dtype, device=total_logits.device)

    valid_batch_indices, valid_aa_indices = zip(*valid_entries)
    valid_batch_indices = torch.tensor(valid_batch_indices, dtype=torch.long, device=total_logits.device)
    valid_aa_indices = torch.tensor(valid_aa_indices, dtype=torch.long, device=total_logits.device)

    logits = softmax(total_logits)
    return logits[valid_batch_indices, valid_aa_indices]

def create_esm_encodings(config, device, train_dataset, val_dataset, test_dataset):
    """
    Generates and saves ESM embeddings for training, validation, and test datasets.
    
    Loads the appropriate model and alphabet based on configuration. Encodes each dataset, runs optional PCA,
    and saves encoded outputs to disk.
    
    Args:
        config: Configuration object with model, dataset, and save parameters.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        train_dataset (pd.DataFrame): Training set.
        val_dataset (pd.DataFrame): Validation set.
        test_dataset (pd.DataFrame): Test set.
    
    Returns:
        tuple: (train_embeddings, val_embeddings, test_embeddings), each an EncodedOutputDataset.
    """

    main_logger.info("Loading ESM model for embeddings...")
    model, alphabet = load_esm_model(config, device)
    save_path = os.path.join(config.dataset.save_path,config.settings.experiment_name)
    
    train_embeddings = generate_embeds_and_save(df=train_dataset,
                                                save_path= save_path,
                                                dataset_filename= config.dataset.train_file_name, 
                                                model= model, 
                                                tokenizer= alphabet, 
                                                device= device,
                                                config= config)
    val_embeddings = generate_embeds_and_save(df=val_dataset,
                                                save_path= save_path,
                                                dataset_filename= config.dataset.val_file_name, 
                                                model= model, 
                                                tokenizer= alphabet, 
                                                device= device,
                                                config= config)
    test_embeddings = generate_embeds_and_save(df=test_dataset,
                                                save_path= save_path,
                                                dataset_filename= config.dataset.test_file_name, 
                                                model= model, 
                                                tokenizer= alphabet, 
                                                device= device,
                                                config= config)

    return train_embeddings, val_embeddings, test_embeddings
