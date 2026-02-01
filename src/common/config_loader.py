import os
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

class TrainingConfig(BaseModel):
    teacher_model: str = "Qwen/Qwen3-VL"
    student_model_name: str = "chandra-distilled"
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_steps: int = 1000
    use_uld: bool = True
    output_dir: str = "./output"

class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self._config: Optional[TrainingConfig] = None
    
    def load_config(self) -> TrainingConfig:
        """
        Load configuration from file or use defaults.
        
        Returns:
            TrainingConfig instance.
        """
        if self._config:
            return self._config

        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f)
                self._config = TrainingConfig(**data)
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {self.config_path}: {e}")
                raise
        else:
            logger.warning("No config file found or provided. Using default configuration.")
            self._config = TrainingConfig()
            
        return self._config

    @staticmethod
    def get_default_config() -> TrainingConfig:
        return TrainingConfig()
