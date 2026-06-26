from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_entrypoint_uses_model_registry():
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "from models.registry import" in train_source
    assert "weather_model" not in train_source


def test_bootstrap_weather_model_module_was_removed():
    assert not (ROOT / "weather_model.py").exists()


def test_baseline_cnn_candidate_is_registered():
    registry_source = (ROOT / "models/registry.py").read_text(encoding="utf-8")

    assert '"baseline_cnn"' in registry_source or "baseline_cnn.NAME" in registry_source
    assert "get_model_candidate" in registry_source


def test_efficientnet_b0_candidate_is_registered_as_default():
    registry_source = (ROOT / "models/registry.py").read_text(encoding="utf-8")
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "efficientnet_b0" in registry_source
    assert 'DEFAULT_MODEL = "efficientnet_b0"' in train_source


def test_efficientnet_b3_candidate_is_registered():
    registry_source = (ROOT / "models/registry.py").read_text(encoding="utf-8")
    model_source = (ROOT / "models/torchvision_candidates.py").read_text(
        encoding="utf-8"
    )

    assert "efficientnet_b3" in registry_source
    assert "EfficientNet_B3_Weights.DEFAULT" in model_source
    assert "EFFICIENTNET_B3_IMAGE_SIZE = 300" in model_source


def test_training_writes_experiment_run_outputs():
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "results/runs" in train_source
    assert "metadata.json" in train_source
    assert "metrics.json" in train_source
    assert "val_predictions.csv" in train_source
    assert "best_val_macro_f1" in train_source
    assert "split_seed" in train_source
    assert "train_seed" in train_source
    assert "num_workers" in train_source
    assert "prefetch_factor" in train_source
    assert "train_seconds" in train_source
    assert "peak_cuda_memory_allocated" in train_source


def test_submission_flow_commands_are_documented():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")

    assert "promote-submission" in justfile
    assert "confirm-submission" in justfile
    assert "promote-submission.py" in scripts_readme
    assert "confirm-submission.py" in scripts_readme


def test_validation_error_audit_command_is_documented():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")

    assert "audit-errors" in justfile
    assert "audit-val-predictions.py" in scripts_readme


def test_stable_error_pool_command_is_documented():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")
    docs_readme = (ROOT / "docs/README.md").read_text(encoding="utf-8")

    assert "stable-error-pool" in justfile
    assert "summarize-val-error-overlap.py" in scripts_readme
    assert "diagnostics/" in docs_readme


def test_validation_image_feature_command_is_documented():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")
    diagnostics_readme = (ROOT / "docs/diagnostics/README.md").read_text(
        encoding="utf-8"
    )

    assert "val-image-features" in justfile
    assert "audit-val-image-features.py" in scripts_readme
    assert "efficientnet-b0-val-image-features" in diagnostics_readme


def test_logit_bias_diagnostic_command_is_documented():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")
    diagnostics_readme = (ROOT / "docs/diagnostics/README.md").read_text(
        encoding="utf-8"
    )

    assert "logit-bias-diagnostic" in justfile
    assert "diagnose-logit-bias.py" in scripts_readme
    assert "efficientnet-b0-logit-bias" in diagnostics_readme
