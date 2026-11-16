"""Shared metadata paths for DARRA-IER label generation, training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_CONFIG = PROJECT_ROOT / "build_soft_labels" / "config.json"


def generated_metadata_paths(split: str, config_path=DEFAULT_LABEL_CONFIG) -> list[str]:
    """Return configured dataset metadata *relative to --data-root*.

    Filenames come from the same JSON configuration as the label generator.
    This intentionally does not fall back to old metadata: such a fallback
    would silently train/evaluate on a different set of labels.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    params = config.get("dataset_params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No dataset_params configured in {config_path}")
    key = "save_train" if split == "train" else "save_test"
    names = []
    for dataset, entry in params.items():
        name = entry.get(key)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{dataset}: missing {key} in {config_path}")
        names.append(name)
    return names


def resolve_metadata_paths(names, data_root):
    """Resolve user-provided or configured metadata locations."""
    root = Path(data_root).expanduser().resolve()
    return [str((root / p).resolve()) if not Path(p).expanduser().is_absolute()
            else str(Path(p).expanduser().resolve()) for p in names]
