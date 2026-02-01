
import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# 1. Mock modal BEFORE importing the script
# We need a custom Mock for App that handles the @app.function decorator
mock_modal = MagicMock()

def mock_function_decorator(**kwargs):
    def decorator(func):
        # Attach .local to the function so we can call it directly in valid modal style
        func.local = func 
        return func
    return decorator

mock_app_instance = MagicMock()
mock_app_instance.function.side_effect = mock_function_decorator
mock_modal.App.return_value = mock_app_instance

sys.modules["modal"] = mock_modal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Now import the script
try:
    from scripts import train_modal
except ImportError:
    # If dependencies are missing, we might fail here. 
    # But since we run in the environment, it should be fine.
    pass

class TestModelLoadingFallback(unittest.TestCase):
    def setUp(self):
        self.dataset_mock = MagicMock()
        self.dataset_mock.__len__.return_value = 10
        
    # We patch the things that the script uses.
    # Note: The script imports them inside the function.
    # So we must patch where they come FROM.
    
    @patch("transformers.AutoProcessor")
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoModelForVision2Seq")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoConfig")
    # We also need to patch datasets and trl interactions if they are used
    # But since we are calling the function, we need to handle the imports inside it.
    # If the environment has them, they will be imported. 
    # We can patch `datasets.load_dataset` etc if we want to mock them.
    @patch("datasets.load_dataset")
    @patch("datasets.Dataset")
    @patch("trl.SFTTrainer") 
    @patch("trl.SFTConfig")
    def test_fallback_logic(
        self, 
        mock_sft_config,
        mock_sft_trainer,
        mock_dataset_cls,
        mock_load_dataset,
        mock_config,
        mock_auto_model,
        mock_vision2seq, 
        mock_causal, 
        mock_img_text, 
        mock_processor
    ):
        """
        Verify that stage1_sft verifies:
        1. AutoModelForImageTextToText (Primary)
        2. AutoModelForCausalLM (Fallback 1)
        3. AutoModelForVision2Seq (Fallback 2)
        4. AutoModel (Last Resort)
        """
        # Setup mocks
        mock_load_dataset.return_value = self.dataset_mock
        mock_dataset_cls.from_list.return_value = self.dataset_mock
        
        # Ensure imports worked
        if not hasattr(train_modal, "stage1_sft"):
            self.fail("Could not import stage1_sft from train_modal")

        # Scenario 1: Primary works
        mock_img_text.from_pretrained.return_value = MagicMock(spec=["config"])
        train_modal.stage1_sft.local(dataset_name="test", max_steps=1)
        mock_img_text.from_pretrained.assert_called()
        mock_causal.from_pretrained.assert_not_called()
        
        # Reset
        mock_img_text.reset_mock()
        mock_causal.reset_mock()
        mock_vision2seq.reset_mock()
        
        # Scenario 2: Primary fails, Causal works
        mock_img_text.from_pretrained.side_effect = ValueError("Not an image text model")
        mock_causal.from_pretrained.return_value = MagicMock(spec=["config"])
        
        train_modal.stage1_sft.local(dataset_name="test", max_steps=1)
        
        mock_img_text.from_pretrained.assert_called()
        mock_causal.from_pretrained.assert_called()
        mock_vision2seq.from_pretrained.assert_not_called()
        
        # Reset
        mock_img_text.reset_mock()
        mock_img_text.from_pretrained.side_effect = ValueError("Not an image text model")
        mock_causal.reset_mock()
        mock_vision2seq.reset_mock()
        
        # Scenario 3: Primary fails, Causal fails, Vision2Seq works
        mock_causal.from_pretrained.side_effect = ValueError("Not a causal model")
        mock_vision2seq.from_pretrained.return_value = MagicMock(spec=["config"])
        
        train_modal.stage1_sft.local(dataset_name="test", max_steps=1)
        
        mock_img_text.from_pretrained.assert_called()
        mock_causal.from_pretrained.assert_called()
        mock_vision2seq.from_pretrained.assert_called()
        
        # Reset
        mock_img_text.reset_mock()
        mock_img_text.from_pretrained.side_effect = ValueError("Not an image text model")
        mock_causal.reset_mock()
        mock_causal.from_pretrained.side_effect = ValueError("Not a causal model")
        mock_vision2seq.reset_mock()
        mock_auto_model.reset_mock()
        
        # Scenario 4: All specific fail, Generic AutoModel works
        mock_vision2seq.from_pretrained.side_effect = ValueError("Unrecognized config")
        mock_auto_model.from_pretrained.return_value = MagicMock(spec=["config"])
        
        train_modal.stage1_sft.local(dataset_name="test", max_steps=1)
        
        mock_vision2seq.from_pretrained.assert_called()
        mock_auto_model.from_pretrained.assert_called()

if __name__ == "__main__":
    unittest.main()
