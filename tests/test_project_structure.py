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


def test_training_writes_experiment_run_outputs():
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "results/runs" in train_source
    assert "metadata.json" in train_source
    assert "metrics.json" in train_source
    assert "val_predictions.csv" in train_source
    assert "best_val_macro_f1" in train_source


def test_submission_flow_commands_are_documented():
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts/README.md").read_text(encoding="utf-8")

    assert "promote-submission" in justfile
    assert "confirm-submission" in justfile
    assert "promote-submission.py" in scripts_readme
    assert "confirm-submission.py" in scripts_readme
