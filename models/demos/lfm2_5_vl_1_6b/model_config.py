# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Tenstorrent

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LFM2_5VLConfig:
    """
    Configuration for LFM2.5-VL-1.6B model.
    
    Based on LiquidAI's LFM2.5-VL architecture with 1.6B parameters.
    Supports variable resolution image processing.
    """
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 5632
    max_position_embeddings: int = 8192
    
    # Vision encoder settings
    vision_hidden_size: int = 1024
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_intermediate_size: int = 4096
    vision_patch_size: int = 14
    vision_image_size: int = 336
    
    # Variable resolution support
    max_image_size: int = 1344
    min_image_size: int = 336
    image_size_step: int = 336
    
    # Normalization
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-5
    
    # Activation
    hidden_act: str = "silu"
    
    # RoPE settings
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # Attention
    attention_dropout: float = 0.0
    
    # TT-specific settings
    device_mesh: Optional[object] = None
    mesh_device: Optional[object] = None
    
    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {
                "type": "linear",
                "factor": 1.0,
            }
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def vision_head_dim(self) -> int:
        return self.vision_hidden_size // self.vision_num_attention_heads
    
    def get_image_size_for_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate the appropriate image size for variable resolution processing.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Tuple of (processed_width, processed_height) aligned to model requirements
        """
        import math
        
        # Calculate aspect ratio preserving dimensions
        max_size = min(self.max_image_size, max(width, height))
        min_size = self.min_image_size
        
        # Round to nearest step
        num_steps = max(1, round(max_size / self.image_size_step))
        target_size = num_steps * self.image_size_step
        
        # Scale both dimensions proportionally
        scale = target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Round to patch size
        patch_size = self.vision_patch_size
        new_width = (new_width // patch_size) * patch_size
        new_height = (new_height // patch_size) * patch_size
        
        return max(min_size, new_width), max(min_size, new_height)