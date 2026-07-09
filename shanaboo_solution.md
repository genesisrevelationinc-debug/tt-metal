```diff
--- /dev/null
+++ b/models/experimental/time_series_transformer/__init__.py
@@ -0,0 +1,6 @@
+# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
+#
+# SPDX-License-Identifier: Apache-2.0
+
+"""Time Series Transformer model implementation using TTNN APIs."""
+
--- /dev/null
+++ b/models/experimental/time_series_transformer/model.py
@@ -0,0 +1,489 @@
+# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
+#
+# SPDX-License-Identifier: Apache-2.0
+
+"""Time Series Transformer model using TTNN APIs."""
+
+import math
+from typing import Dict, List, Optional, Tuple, Union
+
+import torch
+import ttnn
+
+
+class TimeSeriesTransformerConfig:
+    """Configuration for Time Series Transformer model."""
+
+    def __init__(
+        self,
+        context_length: int = 48,
+        prediction_length: int = 24,
+        d_model: int = 64,
+        encoder_layers: int = 3,
+        decoder_layers: int = 3,
+        encoder_attention_heads: int = 4,
+        decoder_attention_heads: int = 4,
+        encoder_ffn_dim: int = 256,
+        decoder_ffn_dim: int = 256,
+        activation_function: str = "gelu",
+        dropout: float = 0.1,
+        attention_dropout: float = 0.1,
+        activation_dropout: float = 0.1,
+        num_static_categorical_features: int = 0,
+        num_static_real_features: int = 0,
+        num_time_features: int = 3,
+        cardinality: List[int] = None,
+        embedding_dimension: List[int] = None,
+        lags_sequence: List[int] = None,
+        scaling: str = "mean",
+        distr_output: str = "student_t",
+        num_parallel_samples: int = 100,
+        dtype: str = "float32",
+    ):
+        self.context_length = context_length
+        self.prediction_length = prediction_length
+        self.d_model = d_model
+        self.encoder_layers = encoder_layers
+        self.decoder_layers = decoder_layers
+        self.encoder_attention_heads = encoder_attention_heads
+        self.decoder_attention_heads = decoder_attention_heads
+        self.encoder_ffn_dim = encoder_ffn_dim
+        self.decoder_ffn_dim = decoder_ffn_dim
+        self.activation_function = activation_function
+        self.dropout = dropout
+        self.attention_dropout = attention_dropout
+        self.activation_dropout = activation_dropout
+        self.num_static_categorical_features = num_static_categorical_features
+        self.num_static_real_features = num_static_real_features
+        self.num_time_features = num_time_features
+        self.cardinality = cardinality or []
+        self.embedding_dimension = embedding_dimension or []
+        self.lags_sequence = lags_sequence or []
+        self.scaling = scaling
+        self.distr_output = distr_output
+        self.num_parallel_samples = num_parallel_samples
+        self.dtype = dtype
+
+
+class TTNNLinear:
+    """Linear layer using TTNN."""
+
+    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: str = "float32"):
+        self.in_features = in_features
+        self.out_features = out_features
+        self.has_bias = bias
+        self.weight = ttnn.create_tensor(
+            torch.empty(out_features, in_features).normal_(mean=0.0, std=0.02),
+            layout=ttnn.TILE_LAYOUT,
+            dtype=get_ttnn_dtype(dtype),
+        )
+        if bias:
+            self.bias = ttnn.create_tensor(
+                torch.zeros(out_features),
+                layout=ttnn.TILE_LAYOUT,
+                dtype=get_ttnn_dtype(dtype),
+            )
+
+    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
+        x = ttnn.linear(x, self.weight, bias=self.bias if self.has_bias else None)
+        return x
+
+
+class TTNNLayerNorm:
+    """Layer normalization using TTNN."""
+
+    def __init__(self, normalized_shape: int, eps: float = 1e-5, dtype: str = "float32"):
+        self.normalized_shape = normalized_shape
+        self.eps = eps
+        self.weight = ttnn.create_tensor(
+            torch.ones(normalized_shape),
+            layout=ttnn.TILE_LAYOUT,
+            dtype=get_ttnn_dtype(dtype),
+        )
+        self.bias = ttnn.create_tensor(
+            torch.zeros(normalized_shape),
+            layout=ttnn.TILE_LAYOUT,
+            dtype=get_ttnn_dtype(dtype),
+        )
+
+    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
+        x = ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
+        return x
+
+
+class TTNNEmbedding:
+    """Embedding layer using TTNN."""
+
+    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: str = "float32"):
+        self.num_embeddings = num_embeddings
+        self.embedding_dim = embedding_dim
+        self.weight = ttnn.create_tensor(
+            torch.empty(num_embeddings, embedding_dim).normal_(mean=0.0, std=0.02),
+            layout=ttnn.TILE_LAYOUT,
+            dtype=get_ttnn_dtype(dtype),
+        )
+
+    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
+        x = ttnn.embedding(x, self.weight)
+        return x
+
+
+def get_ttnn_dtype(dtype_str: str):
+    """Convert string dtype to ttnn dtype."""
+    mapping = {
+        "float32": ttnn.float32,
+        "bfloat16": ttnn.bfloat16,
