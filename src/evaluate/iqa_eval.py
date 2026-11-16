"""Image-quality evaluation with the DARRA-IER scoring model."""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from tqdm import tqdm

from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.mm_utils import get_model_name_from_path, tokenizer_image_token
from src.model.builder import load_pretrained_model

LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]
LEVEL_WEIGHTS = [5.0, 4.0, 3.0, 2.0, 1.0]


def load_image(path: str) -> Image.Image:
    if path.startswith(("http://", "https://")):
        response = requests.get(path, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(path).convert("RGB")


def pad_square(image: Image.Image, color):
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    result = Image.new(image.mode, (side, side), color)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


def get_level_token_ids(tokenizer, names):
    encoded = [tokenizer(x, add_special_tokens=False).input_ids for x in names]
    if not all(len(ids) == 1 for ids in encoded):
        raise ValueError(f"Quality words must map to exactly one token: {dict(zip(names, encoded))}")
    return [ids[0] for ids in encoded]


def get_prompt(tokenizer, device):
    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], "How would you rate the quality of this image?\n" + DEFAULT_IMAGE_TOKEN)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the image is"
    return tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)


def process_meta(meta_path, args, model, processor, input_ids, level_ids):
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list):
        raise TypeError(f"IQA meta file must be a JSON list: {meta_path}")

    out_path = Path(args.save_dir) / (Path(meta_path).stem + ".jsonl")
    seen = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("image") is not None:
                        seen.add(item["image"])
                except (ValueError, TypeError):
                    pass
    device = torch.device(args.device)
    dtype = next(model.parameters()).dtype
    images, records = [], []
    count = 0
    bg = tuple(round(x * 255) for x in processor.image_mean)
    weights = torch.tensor(LEVEL_WEIGHTS, dtype=torch.float32, device=device)

    def flush():
        nonlocal count
        if not images:
            return
        with torch.inference_mode():
            logits = model(input_ids=input_ids.expand(len(images), -1),
                           images=torch.cat(images, dim=0))["logits"][:, -1, :]
            level_logits = logits[:, level_ids].float()
            closed_probs = torch.softmax(level_logits, dim=-1)
            scores = (closed_probs @ weights).tolist()
            if args.with_prob:
                open_probs = torch.softmax(logits.float(), dim=-1)[:, level_ids].tolist()
            with out_path.open("a", encoding="utf-8") as f:
                for j, sample in enumerate(records):
                    item = {
                        "id": sample["id"], "image": sample["image"],
                        "gt_score": sample.get("gt_score"),
                        "gt_score_norm": sample.get("gt_score_norm"),
                        "logits": dict(zip(args.level_names, level_logits[j].tolist())),
                        "quality_score": scores[j],
                    }
                    if args.with_prob:
                        # Keep official open-vocabulary probability semantics.
                        item["probs"] = dict(zip(args.level_names, open_probs[j]))
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    seen.add(sample["image"])
        count += len(images)
        images.clear()
        records.clear()

    for index, source in enumerate(tqdm(metadata, desc=Path(meta_path).stem)):
        if not isinstance(source, dict):
            raise ValueError(f"Non-object metadata at index {index} in {meta_path}")
        image_path = source.get("image") or source.get("img_path")
        if not image_path:
            print(f"WARNING: missing image field in entry {index} of {meta_path}")
            continue
        if image_path in seen or any(sample["image"] == image_path for sample in records):
            continue
        full_path = (str(Path(args.root_dir) / image_path)
                     if not image_path.startswith(("http://", "https://")) else image_path)
        try:
            image = load_image(full_path)
            if args.image_aspect_ratio == "pad":
                image = pad_square(image, bg)
            tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"]
            images.append(tensor.to(device=device, dtype=dtype))
            records.append({**source, "image": image_path, "id": source.get("id", index)})
        except (OSError, requests.RequestException, ValueError) as error:
            print(f"WARNING: unable to process {image_path}: {error}")
            continue
        if len(images) >= args.batch_size:
            flush()
    # Flush the remainder even if the final metadata images were skipped.
    flush()
    print(f"{Path(meta_path).name}: newly scored={count}, resumed={len(seen)}, output={out_path}")
    return count


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path=args.model_path, model_base=args.model_base,
        model_name=get_model_name_from_path(args.model_path),
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
        device=args.device, preprocessor_path=args.preprocessor_path,
        use_darra=args.use_darra,
    )
    ids = get_level_token_ids(tokenizer, args.level_names)
    prompt_ids = get_prompt(tokenizer, args.device)
    for path in args.meta_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing evaluation metadata: {path}")
        process_meta(path, args, model, processor, prompt_ids, ids)


def parse_args():
    parser = argparse.ArgumentParser(description="DARRA-IER distributional IQA scoring")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--preprocessor-path", default=None)
    parser.add_argument("--meta-paths", required=True, nargs="+")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--save-dir", default="results/darra")
    parser.add_argument("--level-names", nargs="+", default=LEVEL_NAMES)
    parser.add_argument("--use-darra", action="store_true")
    parser.add_argument("--with-prob", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", choices=["pad", "square"], default="pad")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
