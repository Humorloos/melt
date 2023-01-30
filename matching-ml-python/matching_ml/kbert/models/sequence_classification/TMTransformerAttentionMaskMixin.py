"""
Mixin for replacing the original get_extended_attention_mask() method with one that supports 2D attention masks, for
the TM-modification.
"""
import torch
from typing import Tuple

class TMTransformerAttentionMaskMixin():
    def get_extended_attention_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device = None
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(1)
        mask = (1.0 - mask) * -10000.0
        return mask
