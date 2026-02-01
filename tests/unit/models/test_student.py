import pytest
import torch
from src.models.student import ChandraModel, ChandraConfig

def test_chandra_init():
    config = ChandraConfig(llm_backbone="Qwen/Qwen2.5-0.5B")
    # Mocking AutoModelForCausalLM to avoid downloading weights during test
    # In a real scenario, we might use a tiny random model or mock
    with pytest.mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_llm = pytest.mock.MagicMock()
        mock_from_pretrained.return_value = mock_llm
        
        model = ChandraModel(config)
        assert model.config.llm_backbone == "Qwen/Qwen2.5-0.5B"
        assert model.llm == mock_llm

@pytest.fixture
def mock_chandra_model():
    config = ChandraConfig(llm_backbone="test-tiny")
    with pytest.mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_llm = pytest.mock.MagicMock()
        mock_llm.return_value = {"logits": torch.randn(1, 10, 100)} # simple mock output
        mock_from_pretrained.return_value = mock_llm
        
        model = ChandraModel(config)
        return model

# Skip actual forward pass test if we don't have a real model loaded, 
# or use unit test with true mocks.
