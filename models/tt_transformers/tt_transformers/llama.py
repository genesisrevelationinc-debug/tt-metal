# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
from typing import List, Optional, Tuple
from models.tt_transformers.tt.common import (
    precompute_freqs,
    precompute_freqs_for_finetune,
    sample,
    HostEmbedding,
    PagedAttentionConfig,
from models.tt_transformers.tt.llama_decoder import TtTransformerBlock
from models.tt_transformers.tt.llama_embedding import TtLlamaEmbedding
from models.tt_transformers.tt.llama_lm_head import TtLlamaLMHead
from models.tt_transformers.tt.model_config import ModelConfig


class TtLlamaModelForGeneration:
        self,
        configuration,
        state_dict,
        model_config: Optional[ModelConfig] = None,
        paged_attention_config: Optional[PagedAttentionConfig] = None,
        page_table=None,
        kv_cache=None,
        self.configuration = configuration
        self.state_dict = state_dict
        self.paged_attention_config = paged_attention_config
        self.model_config = model_config or ModelConfig()
        self.page_table = page_table
        self.kv_cache = kv_cache
        self.current = 0
        self.mesh_device = self.configuration.mesh_device

        # Precompute freqs
        if hasattr(configuration, "rope_scaling") and configuration.rope_scaling is not None:
            self.freqs = precompute_freqs_for_finetune(configuration)
        else:
            self.freqs = precompute_freqs(configuration)

        # Embedding
        self.tt_model_name = "tt_llama_model"
        # Transformer blocks
        self.transformer_blocks = [
            TtTransformerBlock(
                model_config=self.model_config,
                configuration=configuration,
                state_dict=state_dict,
                layer_num=i,

        # LM Head
        self.lm_head = TtLlamaLMHead(
            model_config=self.model_config,
            configuration=configuration,
            state_dict=state_dict,
            state_dict_prefix="",
    def forward(
        self,
        tokens: ttnn.Tensor,
        mask: Optional[ttnn.Tensor] = None,
        current_pos: int = 0,
        rot_mat=None,
        transformation_mat=None,
        for block in self.transformer_blocks:
            h = block(
                h,
                mask=mask,
                current_pos=current_pos,
                rot_mat=rot_mat,
                transformation_mat=transformation_mat,
        # LM head
        logits = self.lm_head(
            h,
            mask=mask,
            return_logits=return_logits,
        )
        return logits
        self,
        tokens: ttnn.Tensor,
        start_pos: int,
        mask: Optional[ttnn.Tensor] = None,
        current_pos: int = 0,
        rot_mat=None,
        transformation_mat=None,
        for block in self.transformer_blocks:
            h = block(
                h,
                mask=mask,
                current_pos=current_pos,
                rot_mat=rot_mat,
                transformation_mat=transformation_mat,
        # LM head
        logits = self.lm_head(
            h,
            mask=mask,
            return_logits=return_logits,
        )
        return logits