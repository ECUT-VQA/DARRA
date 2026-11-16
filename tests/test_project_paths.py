"""Configuration consistency tests for dataset preparation and model selection."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _label_module():
    spec = importlib.util.spec_from_file_location(
        "darra_soft_label_generator", ROOT / "build_soft_labels" / "gen_soft_label.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_config_uses_one_data_root():
    config = ROOT / "build_soft_labels" / "config.json"
    cfg = json.loads(config.read_text(encoding="utf-8"))
    root = (config.parent / cfg["data_root"]).resolve()
    assert root == ROOT / "Data-DARRA"
    module = _label_module()
    for dataset in cfg["dataset_params"].values():
        expanded = module.resolve_dataset_paths(dataset, root)
        for key in ("split_json", "mos_json", "save_train", "save_test"):
            assert Path(expanded[key]).is_relative_to(root)


def test_label_config_accepts_an_external_data_root(tmp_path):
    module = _label_module()
    example = {
        "split_json": "KONIQ/metas/split.json",
        "mos_json": "KONIQ/metas/mos.json",
        "save_train": "KONIQ/metas/train.json",
        "save_test": "KONIQ/metas/test.json",
        "img_dir": "KONIQ/images",
    }
    expanded = module.resolve_dataset_paths(example, tmp_path)
    assert Path(expanded["mos_json"]) == tmp_path / "KONIQ/metas/mos.json"
    assert expanded["img_dir"] == example["img_dir"]
    assert example["mos_json"] == "KONIQ/metas/mos.json"
