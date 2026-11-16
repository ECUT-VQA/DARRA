"""Single-GPU DARRA-IER evaluation launcher."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.data_config import DEFAULT_LABEL_CONFIG, generated_metadata_paths

ROOT = Path(__file__).resolve().parent
EXTRA_TEST_METAS = (
    "PIPAL/metas/test_pipal_5k.json",
    "LIVE-WILD/metas/test_livew_1k.json",
    "AGIQA3K/metas/test_agiqa_3k.json",
    "TID2013/metas/test_tid2013_3k.json",
    "CSIQ/metas/test_csiq_866.json",
)


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="DARRA-IER image-quality evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/DARRA-IER")
    parser.add_argument("--model-base", default="ModelZoo/mplug-owl2-llama2-7b")
    parser.add_argument("--preprocessor-path", default="preprocessor")
    parser.add_argument("--data-root", default="Data-DARRA")
    parser.add_argument("--label-config", default=str(DEFAULT_LABEL_CONFIG),
                        help="Soft-label configuration used for default evaluation metadata paths")
    parser.add_argument("--meta-paths", nargs="+", default=None,
                        help="Override default metadata paths, relative to --data-root or absolute")
    parser.add_argument("--save-dir", default="results/DARRA-IER")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--with-prob", action="store_true")
    args = parser.parse_args()

    checkpoint = resolve(args.checkpoint)
    model_base = resolve(args.model_base)
    processor = resolve(args.preprocessor_path)
    data_root = resolve(args.data_root)
    output = resolve(args.save_dir)
    if args.batch_size < 1 or args.gpu_id < 0:
        parser.error("GPU ID must be nonnegative and batch size must be positive")
    for label, item in (("checkpoint", checkpoint), ("base weights", model_base),
                        ("preprocessor", processor), ("dataset root", data_root)):
        if not item.is_dir():
            parser.error(f"Missing {label}: {item}")
    for filename in ("darra_config.json", "non_lora_trainables.bin", "adapter_config.json"):
        if not (checkpoint / filename).is_file():
            parser.error(f"Missing trained checkpoint file: {checkpoint / filename}")
    defaults = generated_metadata_paths("test", args.label_config) + list(EXTRA_TEST_METAS)
    meta_paths = []
    for name in args.meta_paths if args.meta_paths is not None else defaults:
        p = Path(name).expanduser()
        p = p if p.is_absolute() else data_root / p
        if p.is_file():
            meta_paths.append(str(p.resolve()))
        elif args.meta_paths:
            parser.error(f"Missing requested metadata file: {p}")
        else:
            print(f"Skipping absent dataset metadata: {p}")
    if not meta_paths:
        parser.error("No evaluation metadata is available")
    command = [sys.executable, "-m", "src.evaluate.iqa_eval",
               "--model-path", str(checkpoint), "--model-base", str(model_base),
               "--preprocessor-path", str(processor), "--root-dir", str(data_root),
               "--save-dir", str(output), "--batch-size", str(args.batch_size),
               "--device", "cuda:0", "--meta-paths", *meta_paths, "--use-darra"]
    if args.with_prob:
        command.append("--with-prob")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
