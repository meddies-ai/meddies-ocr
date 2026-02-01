import torch
import pytest
from src.distillation.uld_loss import UDLLoss

def test_uld_loss_forward():
    loss_fn = UDLLoss()
    student_logits = torch.randn(2, 5, 100) # [Batch, Seq, Vocab]
    teacher_logits = torch.randn(2, 5, 100)
    
    loss = loss_fn(student_logits, teacher_logits)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # scalar
