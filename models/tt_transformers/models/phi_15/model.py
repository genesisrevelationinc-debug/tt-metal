# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import tt_lib as ttl
from typing import Optional, Tuple, Union
from models.tt_transformers.models.llama.model import TtLlamaModel
from models.tt_transformers.models.common import (
    TtModelArgs,
    TtTransformerBlock,
    TtRMSNorm,
    TtEmbedding,
    TtLinear,
)


class TtPhi15ModelArgs(TtModelArgs):
    """Phi-1.5 model configuration"""
    
    def __init__(self, 
                 dim: int = 2048,
                 n_layers: int = 24,
                 n_heads: int = 32,
                 n_kv_heads: int = 32,
                 vocab_size: int = 51200,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 norm_eps: float = 1e-5,
                 rope_theta: float = 10000.0,
                 max_batch_size: int = 32,
                 max_seq_len: int = 2048,
                 **kwargs):
        super().__init__(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            **kwargs
        )


class TtPhi15TransformerBlock(TtTransformerBlock):
    """Phi-1.5 transformer block with specific adaptations"""
    
    def __init__(self, args: TtPhi15ModelArgs, device):
        super().__init__(args, device)
        # Phi-1.5 specific adaptations can be added here
        
    def forward(self, 
                x: ttl.tensor.Tensor,
                start_pos: int,
                freqs_cis: ttl.tensor.Tensor,
                mask: Optional[ttl.tensor.Tensor] = None) -> ttl.tensor.Tensor:
        return super().forward(x, start_pos, freqs_cis, mask)


class TtPhi15Model(nn.Module):
    """Phi-1.5 model implementation using tt-transformers base"""
    
    def __init__(self, 
                 device,
                 args: TtPhi15ModelArgs,
                 weight_dict: dict,
                 dtype: ttl.tensor.DataType = ttl.tensor.DataType.BFLOAT16,
                 state_dict_path: str = None):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = dtype
        
        # Initialize embedding layer
        self.tok_embeddings = TtEmbedding(
            device=device,
            embedding_dim=args.dim,
            vocab_size=args.vocab_size,
            weight_dict=weight_dict,
            prefix="transformer.wte",
            dtype=dtype,
        )
        
        # Initialize transformer blocks
        self.layers = nn.ModuleList([
            TtPhi15TransformerBlock(args, device) 
            for _ in range(args.n_layers)
        ])
        
        # Initialize normalization
        self.norm = TtRMSNorm(
            device=device,
            dim=args.dim,
            weight_dict=weight_dict,
            prefix="transformer.ln",
            dtype=dtype,
        )
        
        # Initialize output head
        self.output = TtLinear(
            device=device,
            weight_dict=weight_dict,
            prefix="lm_head",
            in_features=args.dim,
            out_features=args.vocab_size,
            dtype=dtype,
        )
        
        # Precompute RoPE frequencies
        self.freqs_cis = TtLlamaModel.precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta
        )
        
    def forward(self,
                tokens: Union[torch.Tensor, ttl.tensor.Tensor],
                start_pos: int,
                top_index: Optional[int] = None,
                freq_cis: Optional[ttl.tensor.Tensor] = None) -> ttl.tensor.Tensor:
        
        if isinstance(tokens, torch.Tensor):
            tokens = ttl.tensor.Tensor(
                tokens,
                self.dtype,
                ttl.tensor.Layout.ROW_MAJOR,
                self.device,
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
            )
        
        seq_len = tokens.get_legacy_shape()[1]
        if freq_cis is None:
            freq_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(self.device)
            
        h = self.tok_embeddings.forward(tokens)
        
        mask = None
        if seq_len > 1:
            mask = TtLlamaModel.build_attention_mask(seq_len, start_pos, self.device)
            
        for layer in self.layers:
            h = layer.forward(h, start_pos, freq_cis, mask)
            
        h = self.norm.forward(h)
        output = self.output.forward(h)
        
        return output