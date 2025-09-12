from __future__ import annotations
import os
import re
from typing import Iterable, List, Optional, Tuple, Dict, Any
import torch
import numpy as np
from base_embeddings import EncodedOutputDataset

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _cache_dir(embedding_type: str) -> str:
    """::
    Fixed cache directory (no config). One subfolder per embedding type.
    """
    base_path = "./embedding_cache"  # <- change if needed
    return os.path.join(base_path, embedding_type)


def _items_dir(embedding_type: str) -> str:
    """Where per-target files live."""
    return os.path.join(_cache_dir(embedding_type), "items")

def _safe_id(target_id: str) -> str:
    """
    Make a filesystem-safe filename from target_id.
    Keeps letters, numbers, dot, underscore, hyphen; replaces everything else with '_'.
    """
    return _SAFE_ID_RE.sub("_", str(target_id))


def _item_path(embedding_type: str, target_id: str) -> str:
    """Per-target file path."""
    return os.path.join(_items_dir(embedding_type), f"{_safe_id(target_id)}.pt")


def has_item(embedding_type: str, target_id: str) -> bool:
    """Return True if a cached file exists for this target."""
    return os.path.exists(_item_path(embedding_type, target_id))


def save_item(
    embedding_type: str,
    target_id: str,
    x: torch.Tensor | np.ndarray | List[float],
    logits: Optional[torch.Tensor | np.ndarray | float] = None,
    label: Optional[int | float] = None,
) -> None:
    """
    Save a single target's data to the cache.

    Args:
        embedding_type: e.g. "ESM2".
        target_id: unique sample id.
        x: embedding vector (1D) or single-row tensor/ndarray.
        logits: optional single logit/score (LLR).
        label: optional label (0/1 or float).
    """
    os.makedirs(_items_dir(embedding_type), exist_ok=True)

    # Ensure tensors (1D)
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x).float()
    elif isinstance(x, list):
        x_t = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        x_t = x.detach().float().cpu()
    else:
        raise TypeError(f"Unsupported type for x: {type(x)}")

    x_t = x_t.view(-1)  # flatten to 1D

    logits_t = None
    if logits is not None:
        if isinstance(logits, (float, int)):
            logits_t = torch.tensor([float(logits)], dtype=torch.float32)
        elif isinstance(logits, np.ndarray):
            logits_t = torch.from_numpy(np.array(logits).reshape(-1)).float()
        elif isinstance(logits, torch.Tensor):
            logits_t = logits.detach().float().cpu().view(-1)
        else:
            raise TypeError(f"Unsupported type for logits: {type(logits)}")

    data = {
        "target_id": str(target_id),
        "x": x_t,                       # (D,)
        "logits": logits_t,             # (1,) or (k,) or None
        "label": None if label is None else float(label),
    }
    torch.save(data, _item_path(embedding_type, target_id))


def load_item(embedding_type: str, target_id: str) -> Dict[str, Any]:
    """
    Load a single target's cached data.

    Returns dict with keys: target_id, x (1D tensor), logits (tensor or None), label (float or None).
    Raises FileNotFoundError if not present.
    """
    path = _item_path(embedding_type, target_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached item for {target_id} at {path}")
    data = torch.load(path, map_location="cpu")
    # Minimal sanity: ensure x is tensor
    if not isinstance(data["x"], torch.Tensor):
        data["x"] = torch.tensor(data["x"], dtype=torch.float32)
    if data.get("logits") is not None and not isinstance(data["logits"], torch.Tensor):
        data["logits"] = torch.tensor(data["logits"], dtype=torch.float32).view(-1)
    return data


def build_dataset_from_ids(
    embedding_type: str,
    target_ids: Iterable[str],
    labels_list: Optional[Iterable[int | float]] = None,
    require_labels: bool = True,
) -> EncodedOutputDataset:
    """
    Assemble an EncodedOutputDataset by loading per-id cached items.

    Args:
        embedding_type: e.g. "ESM2".
        target_ids: iterable of ids (order is preserved).
        labels_list: optional iterable of labels to override cached labels.
        require_labels: if True, raise if any label is missing.

    Returns:
        EncodedOutputDataset(x, label, logits, target_id)
        - x: (N, D)
        - label: (N,) tensor (float) or None (if require_labels=False and labels unavailable)
        - logits: (N, 1) tensor if all present, else None
        - target_id: list[str] length N
    """
    ids = list(map(str, target_ids))
    override_labels = None if labels_list is None else list(labels_list)

    xs: List[torch.Tensor] = []
    ys: List[float] = []
    logits_list: List[Optional[torch.Tensor]] = []
    missing_labels = False

    for i, tid in enumerate(ids):
        item = load_item(embedding_type, tid)
        xs.append(item["x"].view(1, -1))

        # labels
        if override_labels is not None:
            ys.append(float(override_labels[i]))
        else:
            if item["label"] is None:
                missing_labels = True
                ys.append(0.0)  # placeholder
            else:
                ys.append(float(item["label"]))

        # logits
        logits_list.append(item.get("logits"))

    X = torch.cat(xs, dim=0)  # (N, D)

    # labels tensor (optional)
    y_tensor = None
    if require_labels:
        if missing_labels and override_labels is None:
            raise ValueError("Some labels are missing and require_labels=True.")
        y_tensor = torch.tensor(ys, dtype=torch.float32)
    else:
        # Only build if at least one label is present (not required)
        if not all(lbl is None for lbl in ys):
            y_tensor = torch.tensor(ys, dtype=torch.float32)

    # logits tensor (optional, only if all present and same length)
    if all(l is not None for l in logits_list):
        # make each (1,) and stack to (N, 1)
        L = torch.stack([l.view(-1)[0] for l in logits_list], dim=0).view(-1, 1)
    else:
        L = None

    return EncodedOutputDataset(
        x=X,
        label=y_tensor,          # can be None for predict
        logits=L,                # can be None
        target_id=ids,
    )