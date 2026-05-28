# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0

import torch
from models.tt_transformers.tt.common import (
    precompute_freqs,
    prepare_rotary_embedding,
    prepare_rotary_embedding_head_dim,
    sample,
    HostProfiler,
    num_to_core,
)
from models.tt_transformers.tt.llama_attention import TtLlamaAttention
from models.tt_transformers.tt.llama_mlp import TtLlamaMLP
from models.tt_transformers.tt.phi_mlp import TtPhiMLP
from models.tt_transformers.tt.llama_decoder import TtLlamaDecoder
from models.tt_transformers.tt.model_config import ModelArgs

class TtLlamaModelForGeneration:
    def __init__(self, model_args, device, mesh_device, cache_path=None, prefetcher_setup=None):
        self.model_args = model_args
        self.is_phi = hasattr(model_args, "is_phi") and model_args.is_phi
        self.device = device
        self.mesh_device = mesh_device
        self.cache_path = cache_path
        self.prefetcher_setup = prefetcher_setup

        # Build model
        if self.is_phi:
            self.model = TtPhiModel(self.model_args, self.device, self.mesh_device, cache_path=self.cache_path)
        else:
            self.model = TtLlamaModel(self.model_args, self.device, self.mesh_device, cache_path=self.cache_path)

        # Load tokenizer
        self.tokenizer = model_args.tokenizer
        self.vocab_size = self.model_args.vocab_size

        # Precompute freqs
        freqs_fn = prepare_rotary_embedding_head_dim if self.is_phi else prepare_rotary_embedding
        if self.is_phi:
            freqs_fn = prepare_rotary_embedding_head_dim
        self.freqs = freqs_fn(
            self.model_args.max_seq_len,
            self.model_args.head_dim,
            self.model_args.rope_theta,
        )

        self.current_length = 0
        self.prefetcher_setup = prefetcher_setup
        self.current_length = 0

    def forward(self, tokens, start_pos, enable_persistent_kernel=True):
        return self.model(tokens, start_pos, self.freqs, enable_persistent_kernel=enable_persistent_kernel)
class TtLlamaModel:
    def __init__(self, model_args, device, mesh_device, cache_path=None):
        self.model_args = model_args
        self.is_phi = hasattr(model_args, "is_phi") and model_args.is_phi
        self.device = device
        self.mesh_device = mesh_device
        self.cache_path = cache_path
        self.layers = []
        for layer_id in range(self.model_args.n_layers):
            layer = TtLlamaDecoder(
                self.device, self.mesh_device, self.model_args, layer_id, cache_path=self.cache_path,
                mlp_cls=TtPhiMLP if self.is_phi else TtLlamaMLP,
                attention_cls=None,  # Use default attention for now
                is_phi=self.is_phi,
            )
            self.layers.append(layer)

        self.norm = TtLlamaRMSNorm(
            device=self.device,
            state_dict=self.state_dict,
            state_dict_prefix=self.model_args.norm_name,
            state_dict_prefix=self.model_args.norm_name,
            weight_cache_path=self.cache_path,
            layer_num=None,
        )

        # Load lm_head
        lm_head_name = self.model_args.lm_head_name
        lm_head_name = self.model_args.lm_head_name
        self.lm_head = TtLlamaMLP(
            device=self.device,
            dim=self.model_args.dim,
            state_dict=self.state_dict,
            weight_cache_path=self.cache_path,
            weight_key=lm_head_name,
            weight_key=lm_head_name,
            bias_key=None,
        )
        self.tok_embeddings = TtLlamaEmbedding(
            device=self.device,
            state_dict=self.state_dict,
            state_dict_prefix=self.model_args.tok_embeddings_name,
            state_dict_prefix=self.model_args.tok_embeddings_name,
            weight_cache_path=self.cache_path,
            args=self.model_args,
        x = self.lm_head(x)

        return x


class TtPhiModel(TtLlamaModel):
    def __init__(self, model_args, device, mesh_device, cache_path=None):
        super().__init__(model_args, device, mesh_device, cache_path=cache_path)


class TtPhiModelForGeneration(TtLlamaModelForGeneration):
    def __init__(self, model_args, device, mesh_device, cache_path=None, prefetcher_setup=None):
        super().__init__(model_args, device, mesh_device, cache_path=cache_path, prefetcher_setup=prefetcher_setup)