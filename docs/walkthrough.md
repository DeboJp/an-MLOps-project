# FinQA Evaluation & Dataset Preparation Walkthrough

A simple MLOps walkthrough of data prep, implementing RAG, LLM Serving, Fine-Tuning, deployment on Ollama (localhost) and AWS SageMaker, and Evaluation on the FinQA (a quantitative financial question answering dataset with natural language + tabular data and DSL) dataset.

In the interest of time and resources simple small scale finetuning and eval operations were carried out on an Apple M1 Pro chip locally.

---

## Dataset Preparation for MLX Fine-Tuning
Implemented data preparation pipeline in `scripts/prepare_mlx_data.py` to ready the FinQA splits for MLX fine-tuning.

### 1. Hybrid Gold + Noisy Padding Architecture
To prevent training signal corruption and serving-time format mismatch, we transitioned to a hybrid approach:
* **Index Decoding**: Decodes the structural keys in `gold_inds` (e.g. `table_1` and `text_22`) to pull the exact raw sentences and table cells.
* **Header-Aware Layout Mapping**: Automatically formats the raw gold table cells using the served `Columns: [headers] -> Row: [row_str]` format.
* **Controlled Noise**: Fills up the remainder of a 5-chunk context budget with randomly sampled non-gold distractors from the same document page.
* **Bias-Free Shuffling**: Shuffles the final chunks so the model does not learn a positional shortcut.

### 2. MLOps Data Quality Safeguards
We also included some basic data safeguards to make the training data clean and aligned:
* **Chat Template Alignment (Structured Messages JSON)**: Instead of saving training samples as raw strings, we write them as a `"messages"` JSON array (system, user, assistant). When the training framework runs, it automatically detects this and applies the target model's native chat template (inserting control tokens like `<|start_header_id|>`, `<|eot_id|>`). This ensures the prompt layout matches the serving template exactly.
* **Leakage & Duplication Verification**: The preprocessor verifies that there is **0% split leakage** (checking that no validation or test questions overlap with training samples) to ensure we don't cheat during benchmarks.
* **Seeded Random Downsampling**: Rather than slicing contiguous rows (which introduces page grouping bias), we downsample the splits to 500 train, 50 val, and 50 test examples using random sampling with a fixed seed (`random.seed(42)`).

### 3. Dataset Stats
We ran the script and generated three splits under the `./mlx_data` directory:
* `mlx_data/train.jsonl` (500 randomly sampled, grounded examples)
* `mlx_data/val.jsonl` (renamed to `valid.jsonl` for MLX framework compatibility)
* `mlx_data/test.jsonl` (50 test examples)

### 4. Execution Verification
The script runs offline with 0 mismatches, executing instantaneously (~31,000+ examples per second) because it does not require running heavy embedding models or live FAISS vector search during data generation.

---

## What Was Done

### 1. Live RAG Hybrid Retriever (`src/retriever.py`)
Built a modular retrieval pipeline using **LangChain**:
* **Semantic Search:** FAISS vector index using `all-MiniLM-L6-v2` sentence-transformer embeddings.
* **Keyword Search:** BM25 (sparse retriever) to match exact dates, percentages, and metrics.
* **Ensemble Blending:** Combined sparse and dense scores using LangChain's `EnsembleRetriever` with a 50/50 split weight.
* **Table Serialization**: Prepend column headers to all table rows (Header-Aware chunking) so the model retains column context during vector search and reasoning.

### 2. Robust Tokenizer / Parser Fix
* **Issue:** Initially, test predictions resulted in `structure error` during evaluation.
* **Cause:** The official evaluation script `program_tokenization` function automatically appends `"EOF"` to the generated token list. Since our custom model already output `"EOF"`, we ended up with duplicate `["EOF", "EOF"]` tokens. Additionally, lack of spacing after commas in LLM outputs prevented correct splits.
* **Fix:** Updated the clean function in `run_baseline_eval.py` to strip `"EOF"` and trailing commas, and pad commas with spaces (replacing `,` with `, `) to ensure splits work reliably.

---

## RAG Analysis: Why Performance Dropped to 0.0%
Evaluating the baseline under RAG (`--mode retrieved`) highlights the exact challenge of retrieval-augmented reasoning:
1. **Noise Distraction:** The retriever correctly pulled the necessary revenue rows, but also retrieved surrounding sentences. The base LLM got distracted and tried to incorporate irrelevant numbers in its calculations.
2. **Formula Confusion:** Extra retrieved rows confused the baseline model into outputting redundant steps.
3. **Parameter Selection:** If the model extracted numbers correctly, they usually got the wrong arithmetric calculations thereafter.

This serves to give enough cause for us to consider **fine-tuning the generator (LLM)** to teach the model how to ignore noisy context and follow strict mathematical formulas deterministically.

---

## Fine-Tuning & Comparative Evaluation Results

### Running the Fine-Tuning Pipeline
The fine-tuning pipeline is at `scripts/train.sh` and contains memory-safety optimizations to run on limited local hardware (such as 16GB MacBook):
* **Examples of Memory Optimizations Performed**:
  * `--batch-size 1` and `--grad-accumulation-steps 4` to preserve an effective batch size of 4 without exceeding memory constraints.
  * `--grad-checkpoint` to trade minor CPU recomputation to drastically reduce active activation memory.
  * `--max-seq-length 512` to cap padding and sequence tokens. (as the avg seq length was about 600 and outliers could make shape look 1k, which would lead to roughly double the mem space allocated and double the memory usage, max-seq-len controls this)
* **Execution Command**:
  ```bash
  chmod +x scripts/train.sh
  ./scripts/train.sh
  ```

### 1. Fusing the LoRA Adapters
Once training is complete, the LoRA adapters must be mathematically baked back into the base 4-bit weights to produce a standalone model directory:
```bash
.venv/bin/python -m mlx_lm.fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path ./adapters \
  --save-path ./fused_model
```

### 2. Local Serving Architecture Setup
* **Baseline Serving via Ollama (Port 11434)**:
  Compile the baseline Ollama model configuration:
  ```bash
  ollama create finqa-llama3.2 -f config/Modelfile
  ```
  Ensure the Ollama application is running in the background.
* **Fine-Tuned Serving via MLX-LM Server (Port 11435)**:
  Bypass the MLX-GGUF serialization layout bug (which fails to convert fused quantized weights into GGUF row-major arrays) by serving the fused Safetensors natively:
  ```bash
  .venv/bin/python -m mlx_lm.server --model ./fused_model --port 11435
  ```

We ran evaluation comparing the baseline model (`finqa-llama3.2`) served on port 11434 and our tuned model (`finqa-llama3.2-tuned`) served on port 11435 on randomly selected test examples under **retrieved mode** (which makes use of the ensemble RAG retriever instead of the gold chunks context used in training):

### Comparative retrieved-mode performance (tiny sample size)

### Key Findings & Insights:
1. **DSL Syntax Alignment**: The baseline model generated standard natural language filler or incorrectly structured programs, resulting in high structure errors (40%). The fine-tuned model successfully learned the exact comma-separated FinQA DSL rules and sequence terminator, dropping structure errors by half (20%).
2. **Noise Resistance**: Even with a limited fine-tuning epoch, the tuned model successfully ignored retrieved distractors in several cases and matched the numbers exactly, boosting execution accuracy.
3. **Precision Preservation**: Serving the model natively via the MLX-LM server at 4-bit precision bypassed the MLX layout export bug.

---

## Gold Chunks Evaluation Mode
We added a new evaluation mode, `gold_chunks`, to be able to measure performance of generator purely if given golden data in context windows.

Augmented the dataTable and produced this.
* **The Mismatch**: Standard `gold` cell outputs natural language summaries, for fine-tuning the generator needed strictly structured row chunks (`Columns: ... -> Row: ...`).
* **The Solution**: `gold_chunks` decodes the gold indices and maps them directly to the Header-Aware Column formatting template, guaranteeing 100% retriever recall while preserving the identical prompt layout used during training.
* **To Run Gold Chunks Evaluation**:
  ```bash
  # Tuned Model
  .venv/bin/python evaluation/run_baseline_eval.py --model finqa-llama3.2-tuned --mode gold_chunks --limit 20 --show-comparison

  # Baseline Model
  .venv/bin/python evaluation/run_baseline_eval.py --model finqa-llama3.2 --mode gold_chunks --limit 20 --show-comparison
  ```

---

## RAG 4-RGB Evaluation Suite
Implemented an evaluation framework based on the **Retrieval-Augmented Generation Benchmark (RGB)** framework to diagnose specific failure modes under realistic, imperfect retrieval contexts.

### 1. The 4 RGB Testbeds on FinQA
Constructed 50-example testbeds from `data/test.json` using the first 50 valid matches (for strict reproducibility, with a `--random` flag available to sample random subsets):

- **Noise Robustness** `data/rgb_eval/noise_robustness.json`: Context is compiled with all gold sentences/rows plus 8 random distracting sentences/rows from the same document page (total $k = 10$ chunks).
- **Negative Rejection** `data/rgb_eval/negative_rejection.json`: Context contains 5 distractors and 0 gold chunks.
  > We enforce a structured refusal token `reject, EOF` rather than free-form refusal text. This prevents downstream parser crashes and leverages the existing DSL tokenizer structure.
- **Information Integration** `data/rgb_eval/information_integration.json`: Curated subset requiring multi-hop synthesis (filtering strictly for examples where `len(gold_inds) >= 2`).
- **Counterfactual Robustness** `data/rgb_eval/counterfactual_robustness.json`: Extracts financial numbers from the expected program, mutates them in the text/table contexts to random alternative values, and computes a new counterfactual `exe_ans` dynamically. This tests if the model is grounded in retrieved facts or relies on internal memory.

### 2. Execution
* **To compile the testbed datasets**:
  ```bash
  .venv/bin/python evaluation/generate_rgb.py
  ```
* **To run evaluations**:
  ```bash
  # Evaluate baseline model
  .venv/bin/python evaluation/run_rgb_eval.py --model finqa-llama3.2
  
  # Evaluate fine-tuned model
  .venv/bin/python evaluation/run_rgb_eval.py --model finqa-llama3.2-tuned
  ```

---

## AWS Serving & Unified Benchmarking
For production environments, the model is served in the cloud on AWS SageMaker real-time GPU instances (`ml.g4dn.xlarge`) using Hugging Face's Text Generation Inference (TGI) Deep Learning Containers.

### 1. Cloud Deployment
The deployment script packages the fused Safetensors model weights, uploads the archive to Amazon S3, and provisions a SageMaker endpoint:
```bash
.venv/bin/python cloud/deploy.py \
  --bucket <your-s3-bucket-name> \
  --role <your-iam-role-arn> \
  --instance-type ml.g4dn.xlarge
```

### 2. Unified Comparative Benchmarking
Once the SageMaker endpoint is active (`InService`), you can run the centralized benchmark runner. It performs pre-flight connectivity checks, detects active endpoints, and evaluates all models (Ollama, MLX, and SageMaker) across standard modes and all 4 RGB testbeds in one sweep, printing a summary table and saving the report to `docs/benchmark_results.md`:
```bash
.venv/bin/python evaluation/run_benchmarks.py --limit 20 --rgb-limit 50
```

### 3. Cleanup to Stop Charges
To avoid ongoing AWS billing, tear down the SageMaker endpoint, model configs, and provisioned infrastructure:
```bash
.venv/bin/python cloud/cleanup.py
```
