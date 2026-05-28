# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0

import torch
from models.tt_transformers.tt.common import (
    TtLlamaAttention,
    TtLlamaMLP,
    TtLlamaRotaryEmbedding,
)
from models.tt_transformers.tt.model_config import TtLlamaModelConfig

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)

        self.self_attn = TtLlamaAttention(config, state_dict, base_url, device, cache_path)
        self.mlp = TtLlamaMLP(config, state_dict, base_url, device, cache_path)
        self.input_layernorm = TtLlamaRMSNorm(
            device, state_dict=state_dict, base_url=f"{base_url}.input_layernorm", config=config
        )
        # Phi-1.5 uses partial rotary embeddings with a factor of 0.5
        if self.partial_rotary_factor < 1.0:
            self.rotary_emb = TtLlamaRotaryEmbedding(
                config, device, state_dict, base_url=f"{base_url}.self_attn.rotary_emb"
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        residual: torch.Tensor,
        layer_past: Optional[Tuple] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.input_layernorm(hidden_states)

            layer_past=layer_past,
            use_cache=use_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_emb=self.rotary_emb,
        )

        hidden_states = residual + hidden_states
        residual: torch.Tensor,
        layer_past: Optional[Tuple] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.post_attention_layernorm(hidden_states)

            layer_past=layer_past,
            use_cache=use_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_emb=None,
        )

        hidden_states = residual + hidden_states