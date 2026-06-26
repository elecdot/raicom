# Models

Model Candidate definitions live here. Register candidates in `registry.py` and keep training orchestration in `train.py`.

The current competition mainline prioritizes TorchVision Model Candidates so the final submission can stay close to the platform runtime dependency set. `efficientnet_b0` remains the default candidate; `efficientnet_b3` is available as the next same-family capacity step. Do not add `timm` to the mainline unless a later experiment proves it is worth the submission and dependency risk.
