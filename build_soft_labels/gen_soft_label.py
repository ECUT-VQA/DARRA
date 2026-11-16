import argparse
import json
import numpy as np
import os
import random
from pathlib import Path
from scipy.stats import norm, pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DARRA-IER distributional quality labels")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Overrides data_root in config.json; relative paths use the current working directory")
    args = parser.parse_args()
    return args


questions = [
    "What do you think about the quality of this image?",
    "Can you rate the quality of this picture?",
    "Can you judge the quality of this image?",
    "How would you rate the quality of this image?",
    "How would you judge the quality of this image?",
    "What is your quality rating for this image?",
    "What's your opinion on the quality of this picture?",
    "Rate the quality of this image.",
    "Could you evaluate the quality of this image?",
    "How do you assess the quality of this image?",
]


def calculate_srcc_plcc(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    plcc, _ = pearsonr(pred, mos)
    return srcc, plcc


def get_level(mos, min_mos, max_mos):
    eps = 1e-8
    texts = ["bad", "poor", "fair", "good", "excellent"]
    for idx in range(1, len(texts) + 1):
        mos_left = min_mos + (idx - 1) / 5 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 5 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            level = idx
            break
    text = texts[level - 1]
    return text


def adjust_gaussian_bar(probs, score):
    """
    alpha * (a + b + c + d + e) + 5 * beta = 1
    alpha * (5a + 4b + 3c + 2d + e) + 15 beta = score
    ==>
    alpha * A + 5 * beta = 1
    alpha * B + 15 * beta = score
    """
    A = np.array(probs).sum()
    B = np.inner(np.array(probs), np.array([5, 4, 3, 2, 1]))
    alpha = (score - 3) / (B - 3. * A + 1e-9)
    beta = (1. - alpha * A) / 5.
    return alpha, beta


def get_binary_probs(mos, min_mos=1.0, max_mos=5.0):
    eps = 1e-8
    probs = [0, 0, 0, 0, 0]
    for idx in range(1, len(probs)):
        mos_left = min_mos + (idx - 1) / 4 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 4 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            probs[idx - 1] = (mos_right - mos) / (mos_right - mos_left)
            probs[idx] = (mos - mos_left) / (mos_right - mos_left)
            break
    assert np.array((np.array(probs) == 0)).sum() == 3
    assert round(np.array(probs).sum(), 5) == 1
    probs = probs[::-1]  # should start with "excellent" & end with "bad"
    return probs


def main(cfg, answer):
    density_type = cfg["density_type"]  # ["pdf", "cdf"]
    thre_std = cfg["thre_std"]
    thre_diff = cfg["thre_diff"]
    with open(cfg["split_json"]) as fr:
        split = json.load(fr)
    with open(cfg["mos_json"]) as fr:
        mos_dict = json.load(fr)
    save_train = cfg["save_train"]
    save_test = cfg["save_test"]
    img_dir = cfg["img_dir"]

    moses, stds, imgs = [], [], []
    for img in mos_dict:
        moses.append(mos_dict[img]["mos"])
        stds.append(mos_dict[img]["std"])
        imgs.append(img)
    if not moses:
        raise ValueError(f"No MOS records in {cfg['mos_json']}")
    max_mos = max(float(value) for value in moses)
    min_mos = min(float(value) for value in moses)
    if not np.isfinite([min_mos, max_mos]).all() or max_mos <= min_mos:
        raise ValueError("MOS range must have finite, distinct minimum and maximum values")

    num_binary, idx = 0, 0
    preds, gts, raw_diffs, diffs, alphas, betas = [], [], [], [], [], []
    train_metas, test_metas = [], []
    for img, mos_str, std_str in zip(imgs, moses, stds):
        mos, std = float(mos_str), float(std_str)
        if os.path.basename(img) in split["train"]:
            training = True
        elif os.path.basename(img) in split["test"]:
            training = False
        else:
            idx += 1
            # print(idx, img)
            continue

        text = get_level(mos, min_mos, max_mos)
        query = random.choice(questions)
        resp = answer.replace("{}", text)

        # norm mos and std
        mos_norm = 4 * (mos - min_mos) / (max_mos - min_mos) + 1  # [0, 1] -> [1, 5]
        std_norm = 4 * std / (max_mos - min_mos)

        if not np.isfinite(std_norm) or std_norm < 0:
            raise ValueError(f"Invalid subjective-score standard deviation for {img}: {std}")
        # ["excellent", "good", "fair", "poor", "bad"] -> [5, 4, 3, 2, 1]
        # For zero deviation, the binary target is exact and avoids Gaussian NaNs.
        probs = []
        for x in range(5, 0, -1):
            if std_norm == 0:
                prob = get_binary_probs(mos_norm)[5 - x]
            elif density_type == "cdf":
                prob = norm.cdf(x + 0.5, mos_norm, std_norm) - norm.cdf(x - 0.5, mos_norm, std_norm)
            elif density_type == "pdf":
                prob = norm.pdf(x, loc=mos_norm, scale=std_norm)
            else:
                raise ValueError(f"Unknown density_type: {density_type}")
            probs.append(prob)

        mos_rec = np.inner(np.array(probs), np.array([5, 4, 3, 2, 1]))
        raw_diff = abs(mos_rec - mos_norm)
        raw_diffs.append(raw_diff)

        alpha, beta = adjust_gaussian_bar(probs, mos_norm)
        probs_norm = [max(float(value * alpha + beta), 0.0) for value in probs]
        total_prob = sum(probs_norm)
        if total_prob > 0 and np.isfinite(total_prob):
            probs_norm = [value / total_prob for value in probs_norm]
        mos_rec = np.inner(np.array(probs_norm), np.array([5, 4, 3, 2, 1]))
        diff = abs(mos_rec - mos_norm)

        if std_norm < thre_std or not np.isfinite(diff) or diff > thre_diff:
            # if std is too small, use binary probs (see Appendix)
            probs_norm = get_binary_probs(mos_norm)
            mos_rec = np.inner(np.array(probs_norm), np.array([5, 4, 3, 2, 1]))
            diff, alpha, beta = abs(mos_rec - mos_norm), 1., 0.
            num_binary += 1

        preds.append(mos_rec)
        gts.append(mos_norm)
        diffs.append(diff)
        alphas.append(alpha)
        betas.append(beta)

        meta = {
            "id": os.path.basename(img) + f"->{mos_str}",
            "image": os.path.join(img_dir, img),
            "gt_score": mos,
            "gt_score_norm": mos_norm,
            "level_probs_org": probs,
            "level_probs": probs_norm,
            "std": std,
            "std_norm": std_norm,
        }
        if training:
            conversations = [
                {
                    "from": "human",
                    "value": query + "\n<|image|>",
                },
                {
                    "from": "gpt",
                    "value": resp,
                },
            ]
            meta["conversations"] = conversations
            train_metas.append(meta)
        else:
            del meta["level_probs_org"]
            del meta["level_probs"]
            test_metas.append(meta)

    print("=" * 100)
    print(f"save {len(train_metas)} into {save_train}")
    Path(save_train).parent.mkdir(parents=True, exist_ok=True)
    with open(save_train, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(train_metas, indent=4))

    print(f"save {len(test_metas)} into {save_test}")
    Path(save_test).parent.mkdir(parents=True, exist_ok=True)
    with open(save_test, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(test_metas, indent=4))

    if not preds:
        raise ValueError(f"No images were matched to the train/test split in {cfg['split_json']}")
    srcc, plcc = calculate_srcc_plcc(preds, gts)
    print("srcc:", srcc, "plcc:", plcc)
    print("[raw_diff]", "l1:", sum(raw_diffs) / len(raw_diffs), "l2:", np.sqrt((np.array(raw_diffs)**2).mean()))
    print("[diff]", "l1:", sum(diffs) / len(diffs), "l2:", np.sqrt((np.array(diffs)**2).mean()))
    print("[alpha]", "mean:", np.mean(alphas), "std:", np.std(alphas))
    print("[beta]", "mean:", np.mean(betas), "std:", np.std(betas))
    print("binary / all:", num_binary, "/", len(train_metas) + len(test_metas))


def resolve_dataset_paths(parameters, data_root):
    """Resolve metadata inputs and outputs against one explicit dataset root."""
    resolved = dict(parameters)
    for key in ("split_json", "mos_json", "save_train", "save_test"):
        value = Path(resolved[key]).expanduser()
        resolved[key] = str((value if value.is_absolute() else data_root / value).resolve())
    return resolved


def run():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as fr:
        cfg = json.load(fr)
    if args.data_root is None:
        configured_root = Path(cfg.get("data_root", "../Data-DARRA")).expanduser()
        data_root = (config_path.parent / configured_root).resolve()
    else:
        data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_root}. Set --data-root to the existing dataset folder."
        )
    for dataset in cfg["dataset_params"]:
        random.seed(131)
        main(resolve_dataset_paths(cfg["dataset_params"][dataset], data_root), cfg["answer"])


if __name__ == "__main__":
    run()
