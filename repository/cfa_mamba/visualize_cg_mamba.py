from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plt = None


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAIN_RE = re.compile(
    rf"Train:\s+\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\[(?P<step>\d+)/(?P<steps>\d+)\].*?"
    rf"loss\s+(?P<loss_val>{FLOAT_RE})\s+\((?P<loss_avg>{FLOAT_RE})\).*?"
    rf"pseudo\s+(?P<pseudo>{FLOAT_RE})\s+cons\s+(?P<cons>{FLOAT_RE})\s+"
    rf"ent\s+(?P<ent>{FLOAT_RE})\s+selected\s+(?P<selected>{FLOAT_RE})"
)
ACC_RE = re.compile(r"Accuracy of the network on(?: the)? TARGET(?: test images)?:\s+(?P<acc>{FLOAT_RE})%")


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_matplotlib():
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as matplotlib_pyplot

        plt = matplotlib_pyplot
    return plt


def setup_plot_style() -> None:
    local_plt = ensure_matplotlib()
    local_plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)


def kl_clip_to_student_np(clip_probs: np.ndarray, student_probs: np.ndarray) -> np.ndarray:
    return (clip_probs * (np.log(np.maximum(clip_probs, 1e-12)) - np.log(np.maximum(student_probs, 1e-12)))).sum(axis=1)


def safe_torch_load(path: str | Path, map_location: str | Any = "cpu", weights_only: bool = True) -> Any:
    import torch

    if not weights_only:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_source_parts(config, args, model) -> None:
    f_state = strip_module_prefix(safe_torch_load(config.MODEL.SOURCE_F))
    model.layers.load_state_dict(f_state, strict=False)
    model.patch_embed.load_state_dict(f_state, strict=False)

    if args.neck != "" and not (args.neck == "bn_relu" and config.MODEL.NECK_DEEP == 0):
        b_state = strip_module_prefix(safe_torch_load(config.MODEL.SOURCE_B))
        model.fusion_blocks.load_state_dict(b_state, strict=False)

    c_state = strip_module_prefix(safe_torch_load(config.MODEL.SOURCE_C))
    model.classifier.load_state_dict(c_state, strict=True)


def load_target_parts(config, args, model) -> bool:
    output_dir = Path(config.OUTPUT)
    f_path = output_dir / "target_F.pt"
    b_path = output_dir / "target_B.pt"
    c_path = output_dir / "target_C.pt"
    gate_path = output_dir / "target_semantic_gate.pt"
    if not f_path.exists() or not c_path.exists() or not gate_path.exists():
        return False

    f_state = strip_module_prefix(safe_torch_load(f_path))
    model.base_model.patch_embed.load_state_dict(f_state, strict=False)
    model.base_model.layers.load_state_dict(f_state, strict=False)

    if b_path.exists() and args.neck != "" and not (args.neck == "bn_relu" and config.MODEL.NECK_DEEP == 0):
        b_state = strip_module_prefix(safe_torch_load(b_path))
        model.base_model.fusion_blocks.load_state_dict(b_state, strict=False)

    c_state = strip_module_prefix(safe_torch_load(c_path))
    model.base_model.classifier.load_state_dict(c_state, strict=True)
    gate_state = strip_module_prefix(safe_torch_load(gate_path))
    model.semantic_gate.load_state_dict(gate_state, strict=True)
    return True


def build_target_eval_loader(config):
    from torch.utils.data import DataLoader

    from datautil.imgdata.imgdataload import ImageDataset
    import datautil.imgdata.util as imgutil

    target_domain = config.img_dataset[config.target_env]
    dataset = ImageDataset(
        config.dataset,
        config.DATA.DATA_PATH,
        target_domain,
        transform=imgutil.img_test(config.dg_aug),
        return_idx=True,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.DATA.TEST_BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        persistent_workers=config.DATA.NUM_WORKERS > 0,
    )
    return dataset, loader


def build_clip_cache_loader(config, args, teacher):
    from torch.utils.data import DataLoader

    from datautil.imgdata.imgdataload import ImageDataset

    target_domain = config.img_dataset[config.target_env]
    dataset = ImageDataset(
        config.dataset,
        config.DATA.DATA_PATH,
        target_domain,
        transform=teacher.preprocess,
        return_idx=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.clip_batch_size,
        shuffle=False,
        num_workers=args.clip_num_workers,
        pin_memory=True,
        persistent_workers=args.clip_num_workers > 0,
    )
    return dataset, loader


def default_clip_cache_path(config, args) -> str:
    cache_name = f"{config.dataset}_{config.domains[config.target_env]}_{args.clip_model.replace('/', '-')}.pt"
    return os.path.join(config.OUTPUT, "clip_cache", cache_name)


def get_clip_cache(config, args, device, class_names: list[str]):
    import torch

    from cfa_mamba.clip_teacher import ClipSemanticTeacher

    cache_path = args.clip_cache or default_clip_cache_path(config, args)
    if os.path.exists(cache_path) and not args.force_clip_cache:
        return safe_torch_load(cache_path)

    if args.no_build_clip_cache:
        raise FileNotFoundError(
            f"CLIP cache not found: {cache_path}. Remove --no_build_clip_cache or pass --clip_cache."
        )

    teacher = ClipSemanticTeacher(
        class_names=class_names,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        backend=args.clip_backend,
        temperature=args.clip_temperature,
        device=device,
    )
    clip_dataset, clip_loader = build_clip_cache_loader(config, args, teacher)
    cache = teacher.build_cache(
        clip_loader,
        dataset_size=len(clip_dataset.x),
        cache_path=cache_path,
        force_recompute=args.force_clip_cache,
        amp=not args.disable_amp,
        logger=None,
    )
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def make_config(args):
    from config import get_config
    from utils.dgutil import img_param_init

    args.target_da = True
    args = img_param_init(args)
    return get_config(args)


def build_export_model(config, args, device):
    from cfa_mamba.model import ClipGuidedSfMamba
    from models import build_model

    base_model = build_model(config).to(device)
    load_source_parts(config, args, base_model)

    if args.model_kind == "source":
        return base_model

    model = ClipGuidedSfMamba(
        base_model,
        num_classes=config.MODEL.NUM_CLASSES,
        gate_hidden_dim=args.gate_hidden_dim or None,
        gate_dropout=args.gate_dropout,
        adapter_type=args.adapter_type,
    ).to(device)

    checkpoint = args.checkpoint or os.path.join(config.OUTPUT, "cfa_mamba_training_state.pt")
    if os.path.exists(checkpoint):
        state = safe_torch_load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(strip_module_prefix(state["model"]), strict=True)
        elif isinstance(state, dict):
            model.load_state_dict(strip_module_prefix(state), strict=False)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    elif not load_target_parts(config, args, model):
        raise FileNotFoundError(
            f"Could not find CFA-Mamba checkpoint at {checkpoint} or target parts in {config.OUTPUT}."
        )
    return model


def pool_feature(feature, channel_first: bool):
    if feature.ndim == 4:
        if channel_first:
            return feature.mean(dim=(2, 3))
        return feature.mean(dim=(1, 2))
    return feature.flatten(1)


def run_export(args) -> None:
    import torch
    from torch.cuda.amp import autocast

    from cfa_mamba.class_names import infer_class_names

    config = make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class_names = infer_class_names(config, args.class_names_file)
    clip_cache = get_clip_cache(config, args, device, class_names)
    _, eval_loader = build_target_eval_loader(config)
    model = build_export_model(config, args, device)
    model.eval()

    logits_all = []
    features_all = []
    labels_all = []
    clip_probs_all = []
    indices_all = []

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].long()
            indices = batch[2].long()
            clip_probs = clip_cache[indices].to(device, non_blocking=True)

            with autocast(enabled=config.AMP_ENABLE and device.type == "cuda"):
                if args.model_kind == "cg":
                    logits, feature = model(images, clip_probs=clip_probs, return_feat=True)
                else:
                    logits, feature = model(images, return_feat=True)

            pooled = pool_feature(feature.detach(), getattr(model, "channel_first", True))
            logits_all.append(logits.detach().cpu().float().numpy())
            features_all.append(pooled.cpu().float().numpy())
            labels_all.append(labels.cpu().numpy())
            clip_probs_all.append(clip_probs.detach().cpu().float().numpy())
            indices_all.append(indices.cpu().numpy())

            if args.print_freq > 0 and step % args.print_freq == 0:
                print(f"Exported batch {step}/{len(eval_loader)}")

    logits = np.concatenate(logits_all, axis=0)
    features = np.concatenate(features_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    clip_probs = np.concatenate(clip_probs_all, axis=0)
    indices = np.concatenate(indices_all, axis=0)

    out_path = Path(args.npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        logits=logits,
        features=features,
        labels=labels,
        clip_probs=clip_probs,
        indices=indices,
        class_names=np.asarray(class_names),
        model_kind=np.asarray(args.model_kind),
        dataset=np.asarray(config.dataset),
        source_domain=np.asarray(config.domains[args.source_env[0]]),
        target_domain=np.asarray(config.domains[config.target_env]),
    )

    student_probs = softmax_np(logits)
    pred = logits.argmax(axis=1)
    accuracy = float((pred == labels).mean())
    metadata = {
        "npz": str(out_path),
        "model_kind": args.model_kind,
        "dataset": config.dataset,
        "source_domain": config.domains[args.source_env[0]],
        "target_domain": config.domains[config.target_env],
        "num_samples": int(labels.shape[0]),
        "num_classes": int(logits.shape[1]),
        "accuracy": accuracy,
        "mean_entropy": float(entropy_np(student_probs).mean()),
        "mean_clip_student_kl": float(kl_clip_to_student_np(clip_probs, student_probs).mean()),
        "clip_selected_ratio": float((clip_probs.max(axis=1) >= args.clip_conf_threshold).mean()),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


def parse_log_file(path: str | Path) -> dict[str, Any]:
    train_records = []
    accuracies = []
    last_epoch = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            train_match = TRAIN_RE.search(line)
            if train_match:
                rec = {key: float(value) for key, value in train_match.groupdict().items()}
                rec["epoch"] = int(rec["epoch"])
                rec["epochs"] = int(rec["epochs"])
                rec["step"] = int(rec["step"])
                rec["steps"] = int(rec["steps"])
                rec["x"] = rec["epoch"] + rec["step"] / max(rec["steps"], 1)
                train_records.append(rec)
                last_epoch = rec["epoch"]
                continue

            acc_match = ACC_RE.search(line)
            if acc_match:
                epoch = last_epoch if last_epoch is not None else len(accuracies)
                accuracies.append({"epoch": epoch, "accuracy": float(acc_match.group("acc")) / 100.0})
    return {"path": str(path), "label": Path(path).parent.name or Path(path).stem, "train": train_records, "acc": accuracies}


def load_eval_npz(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    result = {key: data[key] for key in data.files}
    result["path"] = path
    return result


def class_names_from_npz(*datasets: dict[str, Any] | None) -> list[str]:
    for dataset in datasets:
        if dataset is not None and "class_names" in dataset:
            return [str(name) for name in dataset["class_names"].tolist()]
    if datasets and datasets[0] is not None:
        num_classes = int(datasets[0]["logits"].shape[1])
        return [str(i) for i in range(num_classes)]
    return []


def fig_path(out_dir: Path, name: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}.{fmt}"


def savefig(path: Path) -> None:
    ensure_matplotlib()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_accuracy_curve(parsed_logs: list[dict[str, Any]], out_dir: Path, fmt: str) -> bool:
    logs = [item for item in parsed_logs if item["acc"]]
    if not logs:
        print("Skip accuracy curve: no accuracy records found.")
        return False

    plt.figure(figsize=(7.2, 4.4))
    for item in logs:
        xs = [rec["epoch"] for rec in item["acc"]]
        ys = [rec["accuracy"] * 100.0 for rec in item["acc"]]
        plt.plot(xs, ys, marker="o", label=item["label"])
    plt.xlabel("Epoch")
    plt.ylabel("Target accuracy (%)")
    plt.title("Target Accuracy Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(fig_path(out_dir, "01_accuracy_curve", fmt))
    return True


def plot_loss_components(parsed_logs: list[dict[str, Any]], out_dir: Path, fmt: str) -> bool:
    source = next((item for item in parsed_logs if item["train"]), None)
    if source is None:
        print("Skip loss component curve: no training records found.")
        return False

    records = source["train"]
    x = np.asarray([rec["x"] for rec in records])
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(x, [rec["loss_avg"] for rec in records], label="total")
    plt.plot(x, [rec["pseudo"] for rec in records], label="pseudo")
    plt.plot(x, [rec["cons"] for rec in records], label="consistency")
    plt.plot(x, [rec["ent"] for rec in records], label="entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(x, [rec["selected"] for rec in records], color="#666666", linestyle="--", label="selected ratio")
    ax2.set_ylabel("Selected ratio")
    ax2.set_ylim(0.0, 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    savefig(fig_path(out_dir, "02_loss_components", fmt))
    return True


def stratified_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if labels.shape[0] <= max_samples:
        return np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)
    labels = labels.astype(int)
    classes = np.unique(labels)
    per_class = max(1, max_samples // max(len(classes), 1))
    chosen = []
    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        take = min(per_class, cls_idx.shape[0])
        chosen.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    if len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(labels.shape[0]), np.asarray(chosen), assume_unique=False)
        fill = min(max_samples - len(chosen), remaining.shape[0])
        if fill > 0:
            chosen.extend(rng.choice(remaining, size=fill, replace=False).tolist())
    chosen = np.asarray(chosen[:max_samples], dtype=int)
    rng.shuffle(chosen)
    return chosen


def preprocess_features(features: np.ndarray, seed: int) -> np.ndarray:
    features = features.astype(np.float32)
    features = features - features.mean(axis=0, keepdims=True)
    features = features / np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    if features.shape[1] > 50 and features.shape[0] > 50:
        from sklearn.decomposition import PCA

        pca_dim = min(50, features.shape[0] - 1, features.shape[1])
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
    return features


def plot_tsne(source_npz: dict[str, Any] | None, cg_npz: dict[str, Any] | None, out_dir: Path, fmt: str, max_samples: int, seed: int) -> bool:
    datasets = []
    if source_npz is not None and "features" in source_npz:
        datasets.append(("Source-only", source_npz))
    if cg_npz is not None and "features" in cg_npz:
        datasets.append(("CFA-Mamba", cg_npz))
    if not datasets:
        print("Skip t-SNE: no feature npz provided.")
        return False

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Skip t-SNE: sklearn is not installed.")
        return False

    if len(datasets) == 2 and datasets[0][1]["labels"].shape == datasets[1][1]["labels"].shape:
        idx = stratified_indices(datasets[1][1]["labels"], max_samples, seed)
        feature_blocks = [item[1]["features"][idx] for item in datasets]
        label_blocks = [item[1]["labels"][idx] for item in datasets]
    else:
        feature_blocks = []
        label_blocks = []
        for _, data in datasets:
            idx = stratified_indices(data["labels"], max_samples, seed)
            feature_blocks.append(data["features"][idx])
            label_blocks.append(data["labels"][idx])

    all_features = preprocess_features(np.concatenate(feature_blocks, axis=0), seed)
    perplexity = min(30, max(5, (all_features.shape[0] - 1) // 3))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(all_features)

    splits = np.cumsum([block.shape[0] for block in feature_blocks[:-1]])
    embedding_blocks = np.split(embedding, splits, axis=0)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 5.0), squeeze=False)
    num_classes = int(max(np.max(block) for block in label_blocks)) + 1
    cmap = "tab20" if num_classes <= 20 else "nipy_spectral"
    for ax, (name, _), emb, labels in zip(axes.ravel(), datasets, embedding_blocks, label_blocks):
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=8, alpha=0.75, cmap=cmap)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    if num_classes <= 20:
        handles, _ = scatter.legend_elements(num=min(num_classes, 20))
        fig.legend(handles, [str(i) for i in range(min(num_classes, 20))], title="Class", loc="center right")
    savefig(fig_path(out_dir, "03_tsne_features", fmt))
    return True


def plot_clip_confidence(data: dict[str, Any] | None, out_dir: Path, fmt: str, threshold: float) -> bool:
    if data is None or "clip_probs" not in data:
        print("Skip CLIP confidence plot: no clip_probs found.")
        return False

    clip_probs = data["clip_probs"]
    labels = data.get("labels")
    confidence = clip_probs.max(axis=1)
    pseudo = clip_probs.argmax(axis=1)
    selected = confidence >= threshold

    fig, axes = plt.subplots(2 if labels is not None else 1, 1, figsize=(7.2, 6.4 if labels is not None else 4.4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    axes[0].hist(confidence, bins=np.linspace(0.0, 1.0, 31), color="#4c78a8", alpha=0.85)
    axes[0].axvline(threshold, color="#e45756", linestyle="--", label=f"threshold={threshold:.2f}")
    axes[0].set_xlabel("CLIP max probability")
    axes[0].set_ylabel("Samples")
    axes[0].set_title(f"CLIP Confidence Distribution (selected={selected.mean() * 100:.1f}%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if labels is not None:
        bins = np.linspace(0.0, 1.0, 11)
        centers = (bins[:-1] + bins[1:]) * 0.5
        bin_ids = np.clip(np.digitize(confidence, bins, right=False) - 1, 0, len(centers) - 1)
        acc = np.full(len(centers), np.nan)
        counts = np.zeros(len(centers), dtype=int)
        correct = pseudo == labels.astype(int)
        for i in range(len(centers)):
            mask = bin_ids == i
            counts[i] = int(mask.sum())
            if mask.any():
                acc[i] = correct[mask].mean() * 100.0
        axes[1].bar(centers, acc, width=0.08, color="#59a14f", alpha=0.9)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_ylim(0.0, 105.0)
        axes[1].set_xlabel("Confidence bin")
        axes[1].set_ylabel("Pseudo-label accuracy (%)")
        axes[1].set_title("CLIP Pseudo-label Accuracy by Confidence")
        axes[1].grid(alpha=0.25)
        for center, value, count in zip(centers, acc, counts):
            if not np.isnan(value) and count > 0:
                axes[1].text(center, min(value + 2.0, 103.0), str(count), ha="center", va="bottom", fontsize=7)

    savefig(fig_path(out_dir, "04_clip_confidence", fmt))
    return True


def per_class_accuracy(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    pred = logits.argmax(axis=1)
    acc = np.full(num_classes, np.nan)
    counts = np.zeros(num_classes, dtype=int)
    for cls in range(num_classes):
        mask = labels.astype(int) == cls
        counts[cls] = int(mask.sum())
        if mask.any():
            acc[cls] = (pred[mask] == cls).mean()
    return acc, counts


def normalized_confusion(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    pred = logits.argmax(axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for label, item in zip(labels.astype(int), pred.astype(int)):
        if 0 <= label < num_classes and 0 <= item < num_classes:
            matrix[label, item] += 1.0
    row_sum = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sum, 1.0)


def plot_class_accuracy_confusion(
    source_npz: dict[str, Any] | None,
    cg_npz: dict[str, Any] | None,
    out_dir: Path,
    fmt: str,
) -> bool:
    data = cg_npz or source_npz
    if data is None or "logits" not in data or "labels" not in data:
        print("Skip class accuracy/confusion: no logits and labels found.")
        return False

    class_names = class_names_from_npz(cg_npz, source_npz)
    num_classes = int(data["logits"].shape[1])
    if not class_names:
        class_names = [str(i) for i in range(num_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3))
    x = np.arange(num_classes)
    width = 0.38
    plotted = False

    if source_npz is not None:
        src_acc, _ = per_class_accuracy(source_npz["logits"], source_npz["labels"], num_classes)
        axes[0].bar(x - width / 2, src_acc * 100.0, width=width, label="Source-only", color="#9ecae9")
        plotted = True
    if cg_npz is not None:
        cg_acc, _ = per_class_accuracy(cg_npz["logits"], cg_npz["labels"], num_classes)
        offset = width / 2 if source_npz is not None else 0.0
        axes[0].bar(x + offset, cg_acc * 100.0, width=width, label="CFA-Mamba", color="#f28e2b")
        plotted = True

    axes[0].set_ylim(0.0, 105.0)
    axes[0].set_ylabel("Class accuracy (%)")
    axes[0].set_title("Per-class Target Accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    if num_classes <= 35:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([name[:16] for name in class_names], rotation=70, ha="right", fontsize=7)
    else:
        axes[0].set_xlabel("Class index")
    if plotted:
        axes[0].legend()

    matrix = normalized_confusion(data["logits"], data["labels"], num_classes)
    im = axes[1].imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Normalized Confusion Matrix ({'CFA-Mamba' if cg_npz is not None else 'Source-only'})")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    if num_classes <= 30:
        axes[1].set_xticks(x)
        axes[1].set_yticks(x)
        axes[1].set_xticklabels([name[:10] for name in class_names], rotation=70, ha="right", fontsize=6)
        axes[1].set_yticklabels([name[:10] for name in class_names], fontsize=6)
    else:
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    savefig(fig_path(out_dir, "05_class_accuracy_confusion", fmt))
    return True


def plot_alignment_entropy(
    parsed_logs: list[dict[str, Any]],
    source_npz: dict[str, Any] | None,
    cg_npz: dict[str, Any] | None,
    out_dir: Path,
    fmt: str,
) -> bool:
    source = next((item for item in parsed_logs if item["train"]), None)
    if source is not None:
        records = source["train"]
        x = np.asarray([rec["x"] for rec in records])
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.plot(x, [rec["cons"] for rec in records], color="#4c78a8", label="CLIP-Mamba KL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Consistency KL")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(x, [rec["ent"] for rec in records], color="#e45756", label="Student entropy")
        ax2.set_ylabel("Entropy")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
        plt.title("Semantic Alignment and Prediction Entropy")
        savefig(fig_path(out_dir, "06_alignment_entropy_curve", fmt))
        return True

    datasets = []
    if source_npz is not None and "clip_probs" in source_npz:
        datasets.append(("Source-only", source_npz))
    if cg_npz is not None and "clip_probs" in cg_npz:
        datasets.append(("CFA-Mamba", cg_npz))
    if not datasets:
        print("Skip alignment/entropy: no logs or npz data found.")
        return False

    labels = []
    kl_values = []
    ent_values = []
    for name, data in datasets:
        student_probs = softmax_np(data["logits"])
        labels.append(name)
        kl_values.append(kl_clip_to_student_np(data["clip_probs"], student_probs))
        ent_values.append(entropy_np(student_probs))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    axes[0].boxplot(kl_values, labels=labels, showfliers=False)
    axes[0].set_title("Final CLIP-Student KL")
    axes[0].set_ylabel("KL")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].boxplot(ent_values, labels=labels, showfliers=False)
    axes[1].set_title("Final Prediction Entropy")
    axes[1].set_ylabel("Entropy")
    axes[1].grid(axis="y", alpha=0.25)
    savefig(fig_path(out_dir, "06_alignment_entropy", fmt))
    return True


def run_plot(args) -> None:
    setup_plot_style()
    out_dir = Path(args.out_dir)
    parsed_logs = [parse_log_file(path) for path in args.log]
    source_npz = load_eval_npz(args.source_npz)
    cg_npz = load_eval_npz(args.cg_npz)
    main_npz = cg_npz or source_npz

    generated = []
    generated.append(plot_accuracy_curve(parsed_logs, out_dir, args.format))
    generated.append(plot_loss_components(parsed_logs, out_dir, args.format))
    generated.append(plot_tsne(source_npz, cg_npz, out_dir, args.format, args.max_tsne_samples, args.seed))
    generated.append(plot_clip_confidence(main_npz, out_dir, args.format, args.clip_conf_threshold))
    generated.append(plot_class_accuracy_confusion(source_npz, cg_npz, out_dir, args.format))
    generated.append(plot_alignment_entropy(parsed_logs, source_npz, cg_npz, out_dir, args.format))
    print(f"Generated {sum(generated)} figure(s) in {out_dir}")


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--accumulation_steps", type=int)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--optim", type=str)

    parser.add_argument("--dataset", type=str, default="office-home", choices=["office", "office-home", "VISDA-C", "domainnet126", "domainnet"])
    parser.add_argument("--target_env", type=int, default=0)
    parser.add_argument("--source_env", type=int, nargs="+", default=[0])
    parser.add_argument("--split_style", type=str, default="strat")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dg_aug", type=str2bool, default=True)
    parser.add_argument(
        "--neck",
        type=str,
        default="bn_relu",
        choices=["", "bn", "bn_relu", "bn_ln_relu", "bn_relu_ln", "bottleneck", "eca", "se", "cbam"],
    )
    parser.add_argument("--fusion_module", type=str, default="ssm")
    parser.add_argument("--lr_factor1", type=float, default=0.1)
    parser.add_argument("--lr_factor2", type=float, default=1.0)
    parser.add_argument("--bg_ratio", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CFA-Mamba visualization toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Run target inference and save logits/features/CLIP probabilities to npz.")
    add_common_config_args(export_parser)
    export_parser.add_argument("--model_kind", choices=["source", "cg"], required=True)
    export_parser.add_argument("--checkpoint", type=str, default="")
    export_parser.add_argument("--npz", type=str, required=True)
    export_parser.add_argument("--adapter_type", type=str, default="scan", choices=["scan", "gate"])
    export_parser.add_argument("--gate_hidden_dim", type=int, default=0)
    export_parser.add_argument("--gate_dropout", type=float, default=0.0)
    export_parser.add_argument("--clip_backend", type=str, default="auto", choices=["auto", "open_clip", "clip"])
    export_parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    export_parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    export_parser.add_argument("--clip_temperature", type=float, default=0.01)
    export_parser.add_argument("--clip_batch_size", type=int, default=128)
    export_parser.add_argument("--clip_num_workers", type=int, default=8)
    export_parser.add_argument("--clip_cache", type=str, default="")
    export_parser.add_argument("--force_clip_cache", action="store_true")
    export_parser.add_argument("--no_build_clip_cache", action="store_true")
    export_parser.add_argument("--clip_conf_threshold", type=float, default=0.55)
    export_parser.add_argument("--class_names_file", type=str, default="")
    export_parser.add_argument("--seed", type=int, default=0)
    export_parser.add_argument("--print_freq", type=int, default=50)
    export_parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Yacs override options, e.g. MODEL.SOURCE_DIR outputs/vmamba_small TRAIN.EPOCHS 15",
    )

    plot_parser = subparsers.add_parser("plot", help="Create the six analysis figures from logs and exported npz files.")
    plot_parser.add_argument("--log", type=str, nargs="*", default=[])
    plot_parser.add_argument("--source_npz", type=str, default="")
    plot_parser.add_argument("--cg_npz", type=str, default="")
    plot_parser.add_argument("--out_dir", type=str, required=True)
    plot_parser.add_argument("--clip_conf_threshold", type=float, default=0.55)
    plot_parser.add_argument("--max_tsne_samples", type=int, default=2000)
    plot_parser.add_argument("--seed", type=int, default=0)
    plot_parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "export":
        run_export(args)
    elif args.command == "plot":
        run_plot(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
