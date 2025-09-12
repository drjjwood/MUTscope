import logging
import torch
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset

from base_embeddings import EncodedOutputDataset, prepare_input_dataloaders
from esm_embeddings import load_esm_model, generate_embeddings_data

# cache helpers (only touched when use_cache=True)
from embedding_cache import (
    has_item, save_item, build_dataset_from_ids
)

main_logger = logging.getLogger(__name__)


def _embedding_type_from_config(config) -> str:
    """Normalize to a stable key used on disk, e.g. 'ESM2', 'ESM1B'."""
    return str(config.settings.embedding_type).upper()

def _randomize_dataset_labels(ds: EncodedOutputDataset, seed: int = 0) -> EncodedOutputDataset:
    """Permute labels in-place for randomized-control runs. Leaves ds.label as-is if None."""
    if ds.label is None:
        return ds
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(ds.label.shape[0], generator=g)
    ds.label = ds.label[perm]
    return ds

def _generate_for_df(
    df: pd.DataFrame,
    device: torch.device,
    config,
    include_features: bool = True,
) -> dict:
    """
    Generate embeddings/logits for all rows in df using your existing ESM path.

    If include_features=False, we drop any 'feature_vector' info so the returned
    X contains only base embeddings (keeps the cache consistent).
    """
    if df.empty:
        return {"x": torch.empty(0), "logits": None, "label": [], "target_id": []}

    df_work = df if include_features else df.drop(columns=["feature_vector"], errors="ignore")
    df_work = df_work.reset_index(drop=True)
    wt_dl, mt_dl = prepare_input_dataloaders(
        df_work,
        batch_size=config.dataset.loader_batch_size,
        num_workers=config.dataset.num_workers,
    )
    model, alphabet = load_esm_model(config, device)
    final_result = generate_embeddings_data(
        wt_dataloader=wt_dl,
        mt_dataloader=mt_dl,
        model=model,
        alphabet=alphabet,
        device=device,
        config=config,
        log_progress=True,
    )
    return final_result


def build_split_dataset_from_cache(
    df: pd.DataFrame,
    labels_col: Optional[str] = "label",
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    config=None,
) -> Dataset:
    """
    If use_cache=True:
      - generate+save ONLY missing ids to cache (features are ignored when caching)
      - assemble the split by loading from cache (base embeddings only)

    If use_cache=False:
      - DO NOT read from or write to cache
      - generate everything fresh (feature vectors will be included if your pipeline provides them)
      - build the dataset directly in memory
    """
    df = df.copy()
    df["target_id"] = df["target_id"].astype(str)
    df = df.reset_index(drop=True)
    embedding_type = _embedding_type_from_config(config)

    # ---------- No-cache path: generate everything fresh, never touch the cache ----------
    if not use_cache:
        main_logger.info("[cache] Disabled → generating all embeddings in-memory (no cache I/O).")
        out = _generate_for_df(df, device=device, config=config, include_features=True)

        # Keep order from the embedding generator explicitly
        ids = [str(t) for t in out["target_id"]]

        # labels: map by id if present
        y = None
        if labels_col and (labels_col in df.columns):
            id_to_label = {str(r["target_id"]): float(r[labels_col]) for _, r in df.iterrows()}
            y = torch.tensor([id_to_label[i] for i in ids], dtype=torch.float32)

        ds = EncodedOutputDataset(
            x=out["x"].float(),
            label=y,                 # may be None for predict
            logits=out["logits"],    # may be None
            target_id=ids,
        )
        return ds

    # ---------- Cache path ----------
    # 1) Find which rows are missing in cache
    missing_mask = ~df["target_id"].map(lambda tid: has_item(embedding_type, tid))
    missing_df = df[missing_mask].copy()
    missing_df = missing_df.reset_index(drop=True)
    # 2) Generate + SAVE only missing items (force base-only embeddings: include_features=False)
    if not missing_df.empty:
        main_logger.info(f"[cache] {len(missing_df)} items missing → generating & saving to cache…")
        out = _generate_for_df(missing_df, device=device, config=config, include_features=False)
        X, L, labels, ids = out["x"], out["logits"], out["label"], out["target_id"]
        for i, tid in enumerate(ids):
            xi = X[i]
            li = L[i] if (L is not None) else None
            # lbl = labels[i] if (labels is not None) else None
            # save_item(embedding_type, str(tid), xi, logits=li, label=lbl)
            save_item(embedding_type, str(tid), xi, logits=li, label=None)

    # 3) Assemble the split from cache (aligned to df order). Optionally override labels.
    require_labels = bool(labels_col and (labels_col in df.columns))
    labels_list = df[labels_col].tolist() if require_labels else None

    ds = build_dataset_from_ids(
        embedding_type,
        df["target_id"].tolist(),
        labels_list=labels_list,
        require_labels=require_labels,
    )
    return ds

def get_encoded_dataset_cached(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    config,
    use_cache: bool = True,
):
    """Replacement for your old get_encoded_dataset with strict cache control."""
    train_ds = build_split_dataset_from_cache(train_df, device=device, use_cache=use_cache, config=config)
    val_ds   = build_split_dataset_from_cache(val_df,   device=device, use_cache=use_cache, config=config)
    test_ds  = build_split_dataset_from_cache(test_df,  device=device, use_cache=use_cache, config=config)

    # Randomized-control: permute TRAIN labels only (validation/test remain untouched)
    if config.settings.randomised_control_run:
        seed = int(getattr(config.settings, "seed", 0))
        train_ds = _randomize_dataset_labels(train_ds, seed=seed)

    return train_ds, val_ds, test_ds


