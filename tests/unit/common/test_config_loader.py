import os
import pytest
import yaml
from src.common.config_loader import ConfigLoader, TrainingConfig

def test_load_default_config():
    loader = ConfigLoader()
    config = loader.load_config()
    assert isinstance(config, TrainingConfig)
    assert config.batch_size == 4

def test_load_custom_config(tmp_path):
    config_data = {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "teacher_model": "TestTeacher"
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    loader = ConfigLoader(str(config_file))
    config = loader.load_config()
    
    assert config.batch_size == 8
    assert config.learning_rate == 1e-4
    assert config.teacher_model == "TestTeacher"

def test_load_nonexistent_config():
    loader = ConfigLoader("nonexistent.yaml")
    # Should handle error or fall back depending on implementation/requirement.
    # Current impl logs error and raises exception if file not found when path is provided?
    # Actually current impl: if path provided and exists -> load. Else -> default with warning.
    # Wait, existing code: "if self.config_path and os.path.exists(self.config_path): ... else: ... default"
    # So if path provided but NOT exists, it falls to default.
    
    config = loader.load_config()
    assert isinstance(config, TrainingConfig)
