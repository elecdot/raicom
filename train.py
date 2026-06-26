import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.registry import available_model_candidates, get_model_candidate

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = REPO_ROOT / "datasets/6a39ed934d7b489daf5f80a4-momodel/train"
DEFAULT_RUNS_DIR = REPO_ROOT / "results/runs"
DEFAULT_MODEL = "baseline_cnn"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a weather classifier.")
    parser.add_argument(
        "--model",
        choices=available_model_candidates(),
        default=DEFAULT_MODEL,
        help="Model Candidate to train.",
    )
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
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


def default_run_id(model_candidate, seed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{model_candidate}-seed{seed}"


def resolve_run_paths(args, candidate):
    run_id = args.run_id or default_run_id(candidate.name, args.seed)
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


def build_loaders(args, candidate, device):
    image_size = args.image_size or candidate.image_size
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    full_set = datasets.ImageFolder(args.train_dir, transform=tf)
    actual_labels = [
        label
        for label, _ in sorted(
            full_set.class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]
    expected_labels = list(candidate.labels)
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"unexpected class order: {actual_labels}; expected {expected_labels}"
        )
    print("class_to_idx:", full_set.class_to_idx)

    n_val = int(len(full_set) * args.val_ratio)
    n_train = len(full_set) - n_val
    train_set, val_set = random_split(
        full_set,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return loss_sum / total, correct / total, macro_f1


def state_dict_snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train(args):
    candidate = get_model_candidate(args.model)
    image_size = args.image_size or candidate.image_size
    run_id, run_dir, artifact_path = resolve_run_paths(args, candidate)
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this training run.")
    print("run_id:", run_id)
    print("model:", candidate.name)
    print("device:", device)
    print("image_size:", image_size)
    print("train_dir:", args.train_dir)
    print("run_dir:", run_dir)
    print("output:", artifact_path)

    train_loader, val_loader = build_loaders(args, candidate, device)
    model = candidate.build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    metrics = []
    best_epoch = None
    best_val_macro_f1 = None
    best_state = None
    final_val_macro_f1 = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        final_val_macro_f1 = val_macro_f1
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
            }
        )
        if best_val_macro_f1 is None or val_macro_f1 > best_val_macro_f1:
            best_epoch = epoch
            best_val_macro_f1 = val_macro_f1
            best_state = state_dict_snapshot(model)
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifact_path)
    write_json(metrics_path, {"epochs": metrics})
    write_json(
        metadata_path,
        {
            "run_id": run_id,
            "model_candidate": candidate.name,
            "seed": args.seed,
            "image_size": image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "train_dir": str(args.train_dir),
            "device": str(device),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "final_val_macro_f1": final_val_macro_f1,
            "artifact": str(artifact_path),
            "metrics": str(metrics_path),
        },
    )
    print("best_epoch:", best_epoch)
    print("best_val_macro_f1:", f"{best_val_macro_f1:.4f}")


if __name__ == "__main__":
    train(parse_args())
