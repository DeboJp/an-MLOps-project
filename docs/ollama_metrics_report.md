# Ollama Performance Analysis Report

This report presents a detailed analysis of local Ollama inference performance metrics extracted from `~/.ollama/logs/server.log` over the length of the project.

---

## 1. Performance Charts

### Chart 1: Throughput Stability (Sequence)
![Throughput Stability](./plot_throughput_stability.png)
* **Analysis**: Plots decoding speed (TPS) across requests. The dashed red trend line indicates overall throughput stability, demonstrating whether Apple Silicon Metal experiences any performance degradation or VRAM swapping during sequential evaluation loops.
* **Takeaway**: Apple Silicon's Unified Memory Architecture (UMA) successfully prevents throughput degradation during batch runs. The stable trend line hovering around ~60 TPS confirms that the model does not suffer from VRAM swapping or thermal throttling over continuous sequential evaluation requests, for our project.

---

### Chart 2: Latency Penalty vs. Input Context Size
![Latency vs Context Size](./plot_latency_vs_context.png)
* **Analysis**: Displays the time taken to process prompts vs. prompt length. The green line shows the linear fit slope (ms per token), demonstrating how the context size (e.g. from retrieved mode noise) adds a pre-fill latency penalty.
* **Takeaway**: Prompt pre-fill latency scales strictly linearly with context size. Every extra retrieved row or text block of noise directly penalizes prompt processing speed, which aligns with our standard beliefs of that token budget constraints are crucial to prevent latency spikes.

---

### Chart 3: Autoregressive Token Generation Scaling
![Generation Scaling](./plot_generation_scaling.png)
* **Analysis**: Illustrates generation time vs. output tokens. The slope represents the average decoding rate per token, while the y-intercept represents the base server overhead for executing requests.
* **Takeaway**: Token generation scales linearly with output length, confirming that decoding throughput remains completely stable without decay as the generation sequence length grows. The tiny y-intercept indicates negligible fixed server pre-flight overhead.

---

### Chart 4: Generation Throughput Speed over Time (Cluster 2 Zoom)
![Generation Speed over Time](./plot_generation_speed_over_time.png)
* **Analysis**: Displays generation TPS mapped chronologically, zoomed in on the active evaluation run (Cluster 2). With the x-axis formatted in seconds, this view captures the steady-state autoregressive decoding speed, showing a highly stable throughput hovering consistently between 52.5 and 58.9 TPS over standard Metal execution threads.
* **Takeaway**: In steady-state operation, decoding throughput is highly consistent with minimal jitter. This level of predictability ensures that user-facing token streaming experiences remain smooth and uninterrupted.

---

### Chart 5: Prompt Evaluation Speed over Time (Cluster 2 Zoom)
![Prompt Speed over Time](./plot_prompt_speed_over_time.png)
* **Analysis**: Plots prompt processing speed over time, zoomed in on the active evaluation run (Cluster 2) at a seconds resolution. The chart reveals that all prompt evaluations are warm prefix-cache hits (ranging from 453.4 to 573.7 TPS with a mean of 514.3 TPS), which could imply that Ollama successfully reused context embeddings and bypassed pre-fill compute overhead.
* **Takeaway**: Once the static system prompt is loaded into Ollama's prefix cache, all subsequent requests in the sequence achieve near-instant pre-fill speeds (averaging 514.3 TPS). Since the chart shows only warm cache hits, it may confirm that Ollama maintains this high pre-fill throughput consistently throughout active sequential execution.
