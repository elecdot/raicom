import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.registry import available_model_candidates, get_model_candidate

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = REPO_ROOT / "datasets/6a39ed934d7b489daf5f80a4-momodel/train"
DEFAULT_RUNS_DIR = REPO_ROOT / "results/runs"
DEFAULT_MODEL = "efficientnet_b0"


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    class_to_idx: dict[str, int]
    train_counts: dict[str, int]
    val_counts: dict[str, int]
    val_paths: list[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a weather classifier.")
    parser.add_argument(
        "--model",
        choices=available_model_candidates(),
        default=DEFAULT_MODEL,
        help="Model Candidate to train.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained weights for candidates that support them.",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--run-id",
        help="Experiment Run identifier. Defaults to timestamp-model-seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override model artifact path. Defaults to results/runs/<run-id>/model.pth.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the selected Model Candidate image size.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override the seed used for the stratified split. Defaults to --seed.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Override the seed used for training randomness. Defaults to --seed.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when --num-workers is greater than 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader worker processes alive across epochs.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adamw",
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "cosine"),
        default="cosine",
        help="Learning-rate scheduler to use.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=7,
        help="Stop after this many epochs without val Macro F1 improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--augmentation",
        choices=("none", "mild"),
        default="mild",
        help="Training augmentation recipe.",
    )
    parser.add_argument(
        "--class-weights",
        choices=("none", "inverse", "inverse-sqrt"),
        default="inverse-sqrt",
        help="Class weighting strategy for CrossEntropyLoss.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Enable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast unless the resolved training device is CUDA.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("--persistent-workers requires --num-workers greater than 0")
    if args.label_smoothing < 0:
        raise ValueError("--label-smoothing must be non-negative")


def resolve_seeds(args):
    train_seed = args.train_seed if args.train_seed is not None else args.seed
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    return train_seed, split_seed


def default_run_id(model_candidate, train_seed, split_seed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if train_seed == split_seed:
        return f"{timestamp}-{model_candidate}-seed{train_seed}"
    return f"{timestamp}-{model_candidate}-trainseed{train_seed}-splitseed{split_seed}"


def resolve_run_paths(args, candidate, train_seed, split_seed):
    run_id = args.run_id or default_run_id(candidate.name, train_seed, split_seed)
    run_dir = DEFAULT_RUNS_DIR / run_id
    artifact_path = args.output or run_dir / "model.pth"
    return run_id, run_dir, artifact_path


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_amp(amp_arg, device):
    if amp_arg is None:
        return device.type == "cuda"
    return bool(amp_arg and device.type == "cuda")


def relative(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interpolation_mode(name):
    modes = {
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
    }
    try:
        return modes[name]
    except KeyError as exc:
        choices = ", ".join(sorted(modes))
        raise ValueError(
            f"unknown interpolation {name!r}; choose one of: {choices}"
        ) from exc


def build_transforms(args, candidate):
    image_size = args.image_size or candidate.image_size
    interpolation = interpolation_mode(candidate.interpolation)
    normalize = transforms.Normalize(mean=candidate.mean, std=candidate.std)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=interpolation),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if args.augmentation == "none":
        train_transform = eval_transform
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=interpolation,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return train_transform, eval_transform


def ordered_labels(image_folder):
    return [
        label
        for label, _ in sorted(
            image_folder.class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]


def count_by_class(targets, indices, labels):
    counts = {label: 0 for label in labels}
    for index in indices:
        counts[labels[targets[index]]] += 1
    return counts


def build_loaders(args, candidate, device, train_seed, split_seed):
    train_transform, eval_transform = build_transforms(args, candidate)
    train_full_set = datasets.ImageFolder(args.train_dir, transform=train_transform)
    eval_full_set = datasets.ImageFolder(args.train_dir, transform=eval_transform)
    actual_labels = ordered_labels(eval_full_set)
    expected_labels = list(candidate.labels)
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"unexpected class order: {actual_labels}; expected {expected_labels}"
        )
    print("class_to_idx:", eval_full_set.class_to_idx)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.val_ratio,
        random_state=split_seed,
    )
    indices = list(range(len(eval_full_set.targets)))
    train_indices, val_indices = next(splitter.split(indices, eval_full_set.targets))
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()

    train_set = Subset(train_full_set, train_indices)
    val_set = Subset(eval_full_set, val_indices)
    pin_memory = device.type == "cuda"
    train_generator = torch.Generator().manual_seed(train_seed)
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs["prefetch_factor"] = args.prefetch_factor
        worker_kwargs["persistent_workers"] = args.persistent_workers
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        **worker_kwargs,
    )
    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=eval_full_set.class_to_idx,
        train_counts=count_by_class(
            eval_full_set.targets, train_indices, actual_labels
        ),
        val_counts=count_by_class(eval_full_set.targets, val_indices, actual_labels),
        val_paths=[relative(eval_full_set.samples[index][0]) for index in val_indices],
    )


def build_class_weights(strategy, train_counts, labels, device):
    if strategy == "none":
        return None, {label: 1.0 for label in labels}

    raw_weights = []
    for label in labels:
        count = train_counts[label]
        if count <= 0:
            raise RuntimeError(f"class {label!r} has no training samples")
        if strategy == "inverse":
            raw_weights.append(1.0 / count)
        elif strategy == "inverse-sqrt":
            raw_weights.append(count**-0.5)
        else:
            raise ValueError(f"unknown class weight strategy: {strategy}")

    mean_weight = sum(raw_weights) / len(raw_weights)
    weights = [weight / mean_weight for weight in raw_weights]
    weight_by_label = dict(zip(labels, weights))
    return torch.tensor(weights, dtype=torch.float32, device=device), weight_by_label


def build_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    raise ValueError(f"unknown scheduler: {args.scheduler}")


def summarize_predictions(y_true, y_pred, labels):
    label_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    per_class = {}
    for index, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=label_ids,
        ).tolist(),
    }


def evaluate(model, loader, criterion, device, amp_enabled, record_logits=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred, logits = [], [], []
    non_blocking = device.type == "cuda"
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x)
                loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            if record_logits:
                logits.extend(out.detach().float().cpu().tolist())

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(loader.dataset.dataset.classes))),
        average="macro",
        zero_division=0,
    )
    result = {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if record_logits:
        result["logits"] = logits
    return result


def state_dict_snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_val_predictions(path, val_paths, eval_result, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = eval_result["logits"]
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    confidences = torch.softmax(logits_tensor, dim=1).max(dim=1).values.tolist()
    fieldnames = [
        "path",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
    ] + [f"logit_{label}" for label in labels]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(val_paths):
            true_index = eval_result["y_true"][index]
            pred_index = eval_result["y_pred"][index]
            row = {
                "path": image_path,
                "true_label": labels[true_index],
                "pred_label": labels[pred_index],
                "correct": int(true_index == pred_index),
                "confidence": confidences[index],
            }
            for label, logit in zip(labels, logits[index]):
                row[f"logit_{label}"] = logit
            writer.writerow(row)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_enabled):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    non_blocking = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x)
            loss = criterion(out, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def train(args):
    run_start = time.perf_counter()
    validate_args(args)
    train_seed, split_seed = resolve_seeds(args)
    set_seed(train_seed)
    candidate = get_model_candidate(args.model)
    image_size = args.image_size or candidate.image_size
    run_id, run_dir, artifact_path = resolve_run_paths(
        args,
        candidate,
        train_seed,
        split_seed,
    )
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    val_predictions_path = run_dir / "val_predictions.csv"
    device = resolve_device(args.device)
    amp_enabled = resolve_amp(args.amp, device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this training run.")

    pretrained = bool(args.pretrained and candidate.pretrained_weights_name)
    print("run_id:", run_id)
    print("model:", candidate.name)
    print("pretrained:", pretrained)
    print("device:", device)
    print("amp_enabled:", amp_enabled)
    print("train_seed:", train_seed)
    print("split_seed:", split_seed)
    print("image_size:", image_size)
    print("train_dir:", args.train_dir)
    print("run_dir:", run_dir)
    print("output:", artifact_path)

    loaders = build_loaders(args, candidate, device, train_seed, split_seed)
    labels = list(candidate.labels)
    class_weight_tensor, class_weights = build_class_weights(
        args.class_weights,
        loaders.train_counts,
        labels,
        device,
    )
    model = candidate.build_model(pretrained=pretrained).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=args.label_smoothing,
    )
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    epoch_metrics = []
    best_epoch = None
    best_val_macro_f1 = None
    best_val_accuracy = None
    best_state = None
    final_val_macro_f1 = None
    epochs_without_improvement = 0

    print("train_counts:", loaders.train_counts)
    print("val_counts:", loaders.val_counts)
    print("class_weights:", class_weights)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            loaders.train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_result = evaluate(
            model,
            loaders.val_loader,
            criterion,
            device,
            amp_enabled,
        )
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        if scheduler is not None:
            scheduler.step()

        final_val_macro_f1 = val_result["macro_f1"]
        epoch_metrics.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_seconds": train_seconds,
                "train_samples_per_second": len(loaders.train_loader.dataset)
                / train_seconds,
                "val_loss": val_result["loss"],
                "val_acc": val_result["accuracy"],
                "val_macro_f1": val_result["macro_f1"],
                "val_seconds": val_seconds,
                "val_samples_per_second": len(loaders.val_loader.dataset) / val_seconds,
                "epoch_seconds": epoch_seconds,
            }
        )
        improved = (
            best_val_macro_f1 is None or val_result["macro_f1"] > best_val_macro_f1
        )
        if improved:
            best_epoch = epoch
            best_val_macro_f1 = val_result["macro_f1"]
            best_val_accuracy = val_result["accuracy"]
            best_state = state_dict_snapshot(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"lr={current_lr:.6g}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_result['loss']:.4f}  "
            f"val_acc={val_result['accuracy']:.4f}  "
            f"val_macro_f1={val_result['macro_f1']:.4f}  "
            f"epoch_seconds={epoch_seconds:.1f}"
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "early stopping:",
                f"no val_macro_f1 improvement for {epochs_without_improvement} epochs",
            )
            break

    model.load_state_dict(best_state)
    best_eval_start = time.perf_counter()
    best_eval = evaluate(
        model,
        loaders.val_loader,
        criterion,
        device,
        amp_enabled,
        record_logits=True,
    )
    best_eval_seconds = time.perf_counter() - best_eval_start
    best_summary = summarize_predictions(
        best_eval["y_true"],
        best_eval["y_pred"],
        labels,
    )
    peak_cuda_memory_allocated = None
    peak_cuda_memory_reserved = None
    if device.type == "cuda":
        peak_cuda_memory_allocated = torch.cuda.max_memory_allocated(device)
        peak_cuda_memory_reserved = torch.cuda.max_memory_reserved(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifact_path)
    write_val_predictions(val_predictions_path, loaders.val_paths, best_eval, labels)
    write_json(
        metrics_path,
        {
            "epochs": epoch_metrics,
            "best_epoch": best_epoch,
            "best": {
                "val_loss": best_eval["loss"],
                "val_acc": best_eval["accuracy"],
                "val_macro_f1": best_eval["macro_f1"],
                "per_class": best_summary["per_class"],
                "confusion_matrix": best_summary["confusion_matrix"],
                "eval_seconds": best_eval_seconds,
            },
        },
    )
    total_seconds = time.perf_counter() - run_start
    write_json(
        metadata_path,
        {
            "run_id": run_id,
            "model_candidate": candidate.name,
            "labels": labels,
            "seed": args.seed,
            "train_seed": train_seed,
            "split_seed": split_seed,
            "image_size": image_size,
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_metrics),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.persistent_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "early_stopping_patience": args.early_stopping_patience,
            "augmentation": args.augmentation,
            "class_weighting": args.class_weights,
            "class_weights": class_weights,
            "label_smoothing": args.label_smoothing,
            "split_strategy": "stratified_shuffle",
            "val_ratio": args.val_ratio,
            "train_counts_by_class": loaders.train_counts,
            "val_counts_by_class": loaders.val_counts,
            "train_dir": str(args.train_dir),
            "device": str(device),
            "amp_requested": "auto" if args.amp is None else bool(args.amp),
            "amp_enabled": amp_enabled,
            "pretrained": pretrained,
            "pretrained_weights_name": candidate.pretrained_weights_name,
            "preprocessing": {
                "mean": candidate.mean,
                "std": candidate.std,
                "interpolation": candidate.interpolation,
            },
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_accuracy": best_val_accuracy,
            "final_val_macro_f1": final_val_macro_f1,
            "total_seconds": total_seconds,
            "peak_cuda_memory_allocated": peak_cuda_memory_allocated,
            "peak_cuda_memory_reserved": peak_cuda_memory_reserved,
            "artifact": str(artifact_path),
            "metrics": str(metrics_path),
            "val_predictions": str(val_predictions_path),
        },
    )
    print("best_epoch:", best_epoch)
    print("best_val_macro_f1:", f"{best_val_macro_f1:.4f}")
    if peak_cuda_memory_allocated is not None:
        print(
            "peak_cuda_memory_allocated_mib:",
            f"{peak_cuda_memory_allocated / 1024 / 1024:.1f}",
        )
    print("val_predictions:", relative(val_predictions_path))


if __name__ == "__main__":
    train(parse_args())
