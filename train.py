import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from weather_model import IMAGE_SIZE, LABELS, WeatherCNN

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = (
    REPO_ROOT / "datasets/6a39ed934d7b489daf5f80a4-momodel/train"
)
DEFAULT_OUTPUT = REPO_ROOT / "results/model_sample.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the baseline weather classifier."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
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


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_loaders(args, device):
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    full_set = datasets.ImageFolder(args.train_dir, transform=tf)
    actual_labels = [label for label, _ in sorted(
        full_set.class_to_idx.items(),
        key=lambda item: item[1],
    )]
    if actual_labels != LABELS:
        raise RuntimeError(
            f"unexpected class order: {actual_labels}; expected {LABELS}"
        )
    print("class_to_idx:", full_set.class_to_idx)

    n_val = int(len(full_set) * args.val_ratio)
    n_train = len(full_set) - n_val
    train_set, val_set = random_split(
        full_set, [n_train, n_val],
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


def train(args):
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this training run.")
    print("device:", device)
    print("train_dir:", args.train_dir)
    print("output:", args.output)

    train_loader, val_loader = build_loaders(args, device)
    model = WeatherCNN(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        val_loss, val_acc, val_macro_f1 = evaluate(
            model, val_loader, criterion, device
        )
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    train(parse_args())
