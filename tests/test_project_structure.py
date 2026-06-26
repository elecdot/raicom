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
