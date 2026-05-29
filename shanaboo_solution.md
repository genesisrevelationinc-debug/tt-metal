```diff
--- a/README.md
+++ b/README.md
@@ -1,3 +1,5 @@
+<div align="center">
+
 <h1>
 
 [Hardware](https://tenstorrent.com/hardware/blackhole) | [Install](./INSTALLING.md) |  [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007) | [Bounty](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)
@@ -5,6 +7,10 @@ The Models team is focused on developing the following models, optimizing them f
 [API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) | [Model Demos](./models/demos/)
 </h3>
 
+</div>
+
+---
+
 </div>
 
 ## Featured Models
@@ -54,6 +58,168 @@ The Models team is focused on developing the following models, optimizing them fo
 </div>
 
+<h3>LLVC (Low-Latency Low-Resource Voice Conversion) Model</h3>
+
+<p>LLVC (Low-Latency Low-Resource Voice Conversion) is a real-time voice conversion model from Koe AI specifically optimized for low latency and CPU efficiency. Key features include:
+<ul>
+<li>Ultra-low latency: Designed for real-time voice conversion with minimal delay</li>
+<li>CPU-optimized: Efficient enough to run on CPU in real-time</li>
+<li>Streaming support: True streaming inference with chunked processing</li>
+<li>High quality: Natural voice conversion while maintaining low latency</li>
+</ul>
+</p>
+
+Stage 1 - Bring-Up:
+<ul>
+<li>Implement LLVC using TTNN APIs (Python)</li>
+<li>Implements the full generation pipeline</li>
+<li>Model runs on either N150 or N300 Tenstorrent hardware with no errors</li>
+<li>Supports both modes:
+<ul>
+<li>Streaming mode: Real-time conversion with chunked processing</li>
+<li>Non-streaming mode: Full-context conversion</li>
+</ul>
+</li>
+<li>Produces valid converted audio output</li>
+<li>Output is verifiable (audio quality assessment, compare with PyTorch reference)</li>
+<li>Achieves baseline throughput target:
+<ul>
+<li>At least 50 tokens/second for decoder generation</li>
+<li>Real-time factor (RTF) < 0.3 for streaming mode</li>
+<li>Latency < 100ms for streaming chunks</li>
+</ul>
+</li>
+<li>Accuracy evaluation:
+<ul>
+<li>Speaker similarity > 70% (cosine similarity)</li>
+<li>Content preservation: WER < 3.0</li>
+<li>Token-level accuracy > 95% against PyTorch reference</li>
+</ul>
+</li>
+<li>Audio quality: Natural prosody with minimal artifacts despite low latency</li>
+<li>Clear instructions for setup and running the model</li>
+</ul>
+
+Stage 2 - Basic Optimizations:
+<ul>
+<li>Use optimal sharded/interleaved memory configs for encoder-decoder layers</li>
+<li>Implement efficient sharding strategy for:
+<ul>
+<li>Lightweight encoder layers</li>
+<li>Decoder layers optimized for streaming</li>
+<li>Convolutional layers with causal padding</li>
+<li>Cached convolution states for streaming</li>
+<li>Optional pitch extraction module</li>
+</ul>
+</li>
+<li>Fuse simple ops where possible (e.g., layer normalization, activation functions)</li>
+<li>Store intermediate activations in L1 where beneficial</li>
+<li>Use recommended TTNN/tt-metal flows for streaming audio models</li>
+<li>Leverage TT library of fused ops for convolution blocks</li>
+<li>Optimize chunk-based processing for streaming</li>
+<li>Efficient state management for causal convolutions</li>
+<li>Optimize vocoder integration</li>
+</ul>
+
+Stage 3 - Deeper Optimization:
+<ul>
+<li>Maximize core counts used per inference</li>
+<li>Implement deeper TT-specific optimizations:
+<ul>
+<li>Ultra-low latency streaming (minimize chunk processing time)</li>
+<li>Efficient causal convolution with state caching</li>
+<li>Flash Attention or equivalent where applicable</li>
+<li>Minimize voice conversion latency per chunk</li>
+<li>Batch processing for multiple concurrent streams</li>
+<li>Pipeline encoder/decoder/vocoder stages</li>
+<li>Optimize memory footprint for edge deployment</li>
+</ul>
+</li>
+<li>Minimize memory and TM (tensor manipulation) overheads</li>
+<li>Explore look-ahead strategies for quality vs latency trade-off</li>
+<li>Document any advanced tuning, known limitations, or trade-offs</li>
+<li>Target stretched goals:
+<ul>
+<li>100+ tokens/second generation speed</li>
+<li>RTF < 0.1 for streaming mode (match CPU performance)</li>
+<li>Latency < 50ms for streaming chunks</li>
+<li>Support for 10+ concurrent streams</li>
+</ul>
+</li>
+</ul>
+
+<h3>Model Implementation Details</h3>
+<p>The model architecture consists of:
+<ul>
+<li>Lightweight encoder: Optimized convolutional layers with reduced complexity</li>
+</ul>
+</p>
+
+<h3>Guidance & Starting Points</h3>
+<ul>
+<li>Use the <a href="https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md">TTNN model bring-up tech report</a> as your primary reference</li>
+<li>Reference <a href="https://github.com/tenstorrent/tt-metal/tree/main/models/demos">streaming audio model patterns in tt-metal</a> for streaming architectures</li>
+<li>Use the <a href="https://github.com/KoeAI/LLVC">official LLVC repository</a> for model architecture details</li>
+<li>Refer to the <a href="https://arxiv.org/abs/2311.