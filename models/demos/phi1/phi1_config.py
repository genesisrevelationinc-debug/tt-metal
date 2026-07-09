# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional


@dataclass
class Phi1Config:
    """Configuration class for Microsoft Phi-1 model."""

    # Model architecture
    vocab_size: int = 51200
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16  # Phi-1 uses MHA, not GQA
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5  # Phi-1 uses partial rotary embeddings
    head_dim: int = 128  # hidden_size // num_attention_heads

    # Activation and normalization
    hidden_act: str = "gelu"  # Phi-1 uses GELU, not SiLU
    layer_norm_eps: float = 1e-5
    use_cache: bool = True

    # Dropout (disabled for inference)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Initialization
    initializer_range: float = 0.02

    # TT-specific configuration
    num_devices: int = 1
    batch_size: int = 1
    max_seq_len: int = 2048
    dtype: str = "bfloat16"

    # Performance tuning
    use_sharding: bool = True
    use_l1_cache: bool = True
    fuse_ops: bool = True

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_name: str = "microsoft/phi-1") -> "Phi1Config":
        """Create config from pretrained model name."""
        config = cls()
        if "phi-1" in model_name:
            # Phi-1 specific overrides
            config.vocab_size = 51200
            config.hidden_size = 2048
            config.intermediate_size = 8192
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.max_position_embeddings = 2048
        return config

    @property
    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        embed_params = self.vocab_size * self.hidden_size
        layer_params = self.num_hidden_layers * (
            4 * self.hidden_size * self.hidden_size +  # Attention QKV + O
            3 * self.hidden_size * self.intermediate_size  # MLP
        )
        return embed_params + layer_params