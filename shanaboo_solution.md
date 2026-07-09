```diff
--- /dev/null
+++ b/models/experimental/time_series_transformer/tt/tt_time_series_transformer.py
@@ -0,0 +1,412 @@
+# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
+#
+# SPDX-License-Identifier: Apache-2.0
+
+"""Time Series Transformer model implementation using TTNN APIs."""
+
+import math
+from typing import Dict, List, Optional, Tuple, Union
+
+import torch
+import ttnn
+
+
+class TTTimeSeriesTransformerConfig:
+    """Configuration for Time Series Transformer model."""
+
+    def __init__(
+        self,
+        prediction_length: int = 24,
+        context_length: int = 168,
+        input_size: int = 1,
+        d_model: int = 64,
+        n_heads: int = 2,
+        num_encoder_layers: int = 2,
+        num_decoder_layers: int = 2,
+        dim_feedforward: int = 256,
+        dropout: float = 0.1,
+        activation: str = "gelu",
+        distribution_output: str = "student_t",
+        lags_seq: Optional[List[int]] = None,
+        num_parallel_samples: int = 100,
+        scaling: str = "mean",
+        num_static_categorical_features: int = 0,
+        cardinality: Optional[List[int]] = None,
+        embedding_dimension: Optional[List[int]] = None,
+        num_static_real_features: int = 0,
+        num_time_features: int = 0,
+    ):
+        self.prediction_length = prediction_length
+        self.context_length = context_length
+        self.input_size = input_size
+        self.d_model = d_model
+        self.n_heads = n_heads
+        self.num_encoder_layers = num_encoder_layers
+        self.num_decoder_layers = num_decoder_layers
+        self.dim_feedforward = dim_feedforward
+        self.dropout = dropout
+        self.activation = activation
+        self.distribution_output = distribution_output
+        self.lags_seq = lags_seq or [1, 2, 3, 4, 5, 6, 7]
+        self.num_parallel_samples = num_parallel_samples
+        self.scaling = scaling
+        self.num_static_categorical_features = num_static_categorical_features
+        self.cardinality = cardinality or []
+        self.embedding_dimension = embedding_dimension or []
+        self.num_static_real_features = num_static_real_features
+        self.num_time_features = num_time_features
+
+
+class TTMultiHeadAttention:
+    """Multi-head attention using TTNN operations."""
+
+    def __init__(self, config: TTTimeSeriesTransformerConfig, device: ttnn.Device):
+        self.config = config
+        self.device = device
+        self.d_model = config.d_model
+        self.n_heads = config.n_heads
+        self.head_dim = self.d_model // self.n_heads
+        self.scale = 1.0 / math.sqrt(self.head_dim)
+
+        # Initialize weights on device
+        self.q_weight = self._create_parameter((self.d_model, self.d_model))
+        self.k_weight = self._create_parameter((self.d_model, self.d_model))
+        self.v_weight = self._create_parameter((self.d_model, self.d_model))
+        self.out_weight = self._create_parameter((self.d_model, self.d_model))
+
+    def _create_parameter(self, shape: Tuple[int, ...]) -> ttnn.Tensor:
+        """Create a TTNN tensor parameter on device."""
+        weight = torch.empty(shape)
+        torch.nn.init.xavier_uniform_(weight)
+        weight_tt = ttnn.from_torch(
+            weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
+        )
+        return weight_tt
+
+    def forward(
+        self,
+        query: ttnn.Tensor,
+        key: ttnn.Tensor,
+        value: ttnn.Tensor,
+        mask: Optional[ttnn.Tensor] = None,
+    ) -> ttnn.Tensor:
+        """Forward pass for multi-head attention."""
+        batch_size, seq_len, _ = query.shape
+
+        # Linear projections
+        q = ttnn.matmul(query, self.q_weight)
+        k = ttnn.matmul(key, self.k_weight)
+        v = ttnn.matmul(value, self.v_weight)
+
+        # Reshape for multi-head: (batch, seq, d_model) -> (batch, n_heads, seq, head_dim)
+        q = ttnn.reshape(q, (batch_size, seq_len, self.n_heads, self.head_dim))
+        q = ttnn.permute(q, (0, 2, 1, 3))
+        k = ttnn.reshape(k, (batch_size, seq_len, self.n_heads, self.head_dim))
+        k = ttnn.permute(k, (0, 2, 1, 3))
+        v = ttnn.reshape(v, (batch_size, seq_len, self.n_heads, self.head_dim))
+        v = ttnn.permute(v, (0, 2, 1, 3))
+
+        # Scaled dot-product attention
+        k_transposed = ttnn.permute(k, (0, 1, 3, 2))
+        attn_weights = ttnn.matmul(q, k_transposed)
+        attn_weights = ttnn.multiply(attn_weights, self.scale)
+
+        if mask is not None:
+            attn_weights = ttnn.add(attn_weights, mask)
+
+        attn_weights = ttnn.softmax(attn_weights, dim=-1)
+
+        # Apply attention to values
+        attn_output = ttnn.matmul(attn_weights, v)
+
+        # Reshape back: (batch, n_heads, seq, head_dim) -> (batch, seq, d_model)
+        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
+        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.d_model))
+
+        # Output projection
+        output