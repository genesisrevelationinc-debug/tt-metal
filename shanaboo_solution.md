```diff
--- a/README.md
+++ b/README.md
@@ -1,15 +1,17 @@
-[![tt-metal CI](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml/badge.svg)](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml)
-[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-metal)
-
-<div align="center">
-
-<h1>
-
-[Hardware](https://tenstorrent.com/hardware/blackhole) | [Install](./INSTALLING.md) |  [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007) | [Bounty $](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)
-
-</h1>
-
-**TT-NN** is a Python & C++ Neural Network OP library.
-
-<h3>
-
-[API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) | [Model Demos](./models/demos/)
-
-</h3>
-
-</div>
-
-## Featured Models
-## Models team is focused on developing the following models, optimizing them for performance, accuracy, and compatibility. Follow each model link for more details.
-
->[!IMPORTANT]
-> For a **full model list** see the **[Model Matrix](https://github.com/tenstorrent/tt-metal/blob/main/models/README.md)**, or visit the **[Developer Hub](https://tenstorrent.com/developers)**.
-
->[!NOTE]
-> Performance Metrics:
-> - Time to First Token (TTFT) measures the time (in milliseconds) it takes to generate the first output token after input is received.
-> - T/S/U (Tokens per Second per User): Represents the throughput of first-token generation after prefill. It is calculated as  to / inter-token latency.
-> - T/S (Tokens per Second): Represents total token throughput, calculated as T/S = T/S/U x batch size.
-> - TP (Tensor Parallel) and TP (Data Parallel): Indicate the parallelization factors across multiple devices.
-> - Reported LLM Performance: Based on an input sequence length of 128 tokens for all models.
-> - Performance Data Source: Metrics were collected using the tt-metal model demos (linked above). Results may vary when using other runtimes such as the vLLM inference server.
-
-### [Llama 3.3 70B (TP=32)](./models/demos/llama3_70b_galaxy)
-| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S | TT-Metalium Release | vLLM Tenstorrent Repo Release |
-|-------|----------|-----------|-------|-----------------|-----|---------------------|-------------------------------|
-| 32    | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy) | 53      | 72.5  | 80              | 2268.8  | [v0.65.0-rc7](https://github.com/tenstorrent/tt-metal/tree/v0.65.0-rc7) | [59be953](https://github.com/tenstorrent/vllm/tree/59be953f2bbd21e227f9ef4b779f545f9c3bf599/tt_metal) |
-
-### [Qwen 2.5 7B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
-| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S  | TT-Metalium Release | vLLM Tenstorrent Repo Release |
-|-------|----------|---------
+<div align="center">
+
+<h1>
+
+[Hardware](https://tenstorrent.com/hardware/blackhole) | [Install](./INSTALLING.md) |  [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007) | [Bounty $](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)
+
+</h1>
+
+**TT-NN** is a Python & C++ Neural Network OP library.
+
+<h3>
+
+[API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) | [Model Demos](./models/demos/)
+
+</h3>
+
+</div>
+
+## Featured Models
+
+The Models team is focused on developing the following models, optimizing them for performance, accuracy, and compatibility. Follow each model link for more details.
+
+>[!IMPORTANT]
+> For a **full model list** see the **[Model Matrix](https://github.com/tenstorrent/tt-metal/blob/main/models/README.md)**, or visit the **[Developer Hub](https://tenstorrent.com/developers)**.
+
+>[!NOTE]
+> Performance Metrics:
+> - Time to First Token (TTFT) measures the time (in milliseconds) it takes to generate the first output token after input is received.
+> - T/S/U (Tokens per Second per User): Represents the throughput of first-token generation after prefill. It is calculated as 1 / inter-token latency.
+> - T/S (Tokens per Second): Represents total token throughput, calculated as T/S = T/S/U x batch size.
+> - TP (Tensor Parallel) and DP (Data Parallel): Indicate the parallelization factors across multiple devices.
+> - Reported LLM Performance: Based on an input sequence length of 128 tokens for all models.
+> - Performance Data Source: Metrics were collected using the tt-metal model demos (linked above). Results may vary when using other runtimes such as the vLLM inference server.
+
+### [LLVC Voice Conversion Model](./models/demos/llvc)
+
+Low-Latency Low-Resource Voice Conversion (LLVC) model implementation using TTNN APIs. This model enables real-time voice conversion with ultra-low latency and high quality audio output.
+
+### [Llama 3.3 70B (TP=32)](./models/demos/llama3_70b_galaxy)
+| Batch | Hardware | TTFT