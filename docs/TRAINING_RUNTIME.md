# Training Runtime

This document defines the runtime boundary for GPU model training and keeps it separate from the platform inference runtime.

## Runtime Split

`requirements.txt` is the momodel platform runtime. It uses CPU PyTorch because the submitted `main.py` must be able to run on CPU during system testing or scoring.

`uv` is the local development runtime. It manages Python 3.9.5 and development tools such as Jupytext, but it should not carry CUDA PyTorch dependencies.

`requirements-train-cu124.txt` is the first GPU training runtime. It is intended for a dedicated Python 3.9.5 environment on a local training machine or a rented cloud GPU instance.

## CUDA Choice

The default training dependency file targets CUDA 12.4 because PyTorch 2.5.1 provides CUDA 12.4 wheels for Python 3.9 and this is a reasonable baseline for RTX 4090 class cloud machines.

If a training machine requires a different CUDA wheel index, add a new file such as `requirements-train-cu121.txt` or `requirements-train-cu126.txt`. Do not replace `requirements.txt` unless the momodel platform runtime changes.

## Setup Shape

Create a dedicated training environment:

```bash
micromamba create -n raicom-train python=3.9.5 pip
micromamba activate raicom-train
python -m pip install -r requirements-train-cu124.txt
```

Verify CUDA before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Run the default Model Candidate training driver from the activated training environment:

```bash
just train
```

The default training command keeps a conservative loader shape: batch size 32
and no DataLoader workers. On a local GPU machine, pass throughput settings
explicitly and compare both runtime and Internal Validation Macro F1 before
adopting them as a repeatable experiment:

```bash
just train --batch-size 32 --num-workers 4 --persistent-workers
```

Training metrics record epoch seconds, images per second, and peak CUDA memory
for later comparison.

Use `--seed` for ordinary runs. When separating split variance from training
variance, set `--split-seed` and `--train-seed` explicitly.

Training artifacts that may be submitted or loaded by `main.py` should be written under `results/`.

## Cloud GPU Notes

Treat cloud GPU machines as training execution venues. Keep the dependency entry point the same: Python 3.9.5 plus one explicit `requirements-train-cu*.txt` file.

When a cloud provider needs extra bootstrap commands, add a small provider-specific script or note rather than folding that setup into the platform `requirements.txt`.
