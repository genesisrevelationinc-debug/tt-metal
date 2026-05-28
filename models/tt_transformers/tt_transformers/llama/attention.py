# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0

import torch
    TtLlamaRotaryEmbedding,
)
from models.tt_transformers.tt.model_config import TtLlamaModelConfig
import math


class TtLlamaAttention(nn.Module):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

            base_url=f"{base_url}.o_proj",
        )

        # For Phi-1.5 partial rotary embeddings
        if self.partial_rotary_factor < 1.0:
            self.rotary_emb = TtLlamaRotaryEmbedding(
                config, device, state_dict, base_url=f"{base_url}.rotary_emb"
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[TtLlamaRotaryEmbedding] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Apply partial rotary embeddings for Phi-1.5
        if rotary_emb is not None or self.rotary_emb is not None:
            rot_emb = rotary_emb if rotary_emb is not None else self.rot_emb
            cos, sin = rot_emb(value_states, seq_len=kv_seq_len)
            # Partial rotary: only apply to first partial_rotary_factor portion of head_dim
            rotary_dim = int(self.head_dim * self.partial_rotary_factor)
            query_rot = query_states[..., :rotary_dim]
            key_rot = key_states[..., :rotary_dim]
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
            query_states = torch.cat([query_rot, query_states[..., rotary_dim:]], dim=-1)
            key_states = torch.cat([key_rot, key_states[..., rotary_dim:]], dim=-1)
        else:
            # Standard full rotary embeddings
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)