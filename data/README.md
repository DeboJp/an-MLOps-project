# FinQA Clean Dataset Splits

This directory contains the cleaned, preprocessed, and compressed splits of the FinQA dataset. The dataset focuses on numerical reasoning over heterogeneous financial data (textual context and structured tables).

## Dataset Files

*   **`train.json`**: The training split used for model fine-tuning (4,420 examples).
*   **`val.json`**: The validation split used for model selection and checkpoint evaluation (447 examples).
*   **`test.json`**: The test split used for baseline and fine-tuned model evaluation (1,147 examples).
*   **`private_test.json`**: The holdout test split containing questions and context but excluding the target programs/answers (used for blind benchmarking).

---

## Schema & Data Structure

Each split is structured as a JSON array of objects. Below is the schema detail for each example:

```json
{
    "id": "ETR/2016/page_23.pdf-2",
    "pre_text": [
        "entergy corporation and subsidiaries management's financial discussion and analysis..."
    ],
    "post_text": [
        "the retail electric price variance is primarily due to..."
    ],
    "table": [
        ["", "Amount (In Millions)"],
        ["2014 net revenue", "$ 5735"],
        ["Retail electric price", "187"],
        ["2015 net revenue", "$ 5829"]
    ],
    "qa": {
        "question": "what is the net change in net revenue during 2015 for entergy corporation?",
        "program": "subtract(5829, 5735)",
        "exe_ans": 94.0,
        "gold_inds": {
            "table_1": "the 2014 net revenue of amount ( in millions ) is $ 5735 ;",
            "table_8": "the 2015 net revenue of amount ( in millions ) is $ 5829 ;"
        }
    }
}
```

### Field Definitions

*   **`id`** *(string)*: Unique identifier of the example.
*   **`pre_text`** *(array of strings)*: Context sentences appearing directly above the table in the financial report.
*   **`post_text`** *(array of strings)*: Context sentences appearing directly below the table.
*   **`table`** *(array of arrays of strings)*: Structured financial table representation where each element is a list representing cells in a row.
*   **`qa`** *(object)*: Core question, program steps, and targets:
    *   **`question`** *(string)*: The question asking for numeric calculations.
    *   **`program`** *(string)*: The ground-truth Domain Specific Language (DSL) sequence of mathematical operations (e.g. `subtract(A, B), divide(#0, B)`).
    *   **`exe_ans`** *(float)*: The numerical value obtained by running the ground-truth program (used for execution accuracy checks).
    *   **`gold_inds`** *(dict)*: The ground-truth gold sentences and rows required to answer the question (used for generator-only testing or training retriever modules).

---

## MLOps Preprocessing Details

In the raw FinQA source, the dataset files are bloated with pre-computed TF-IDF/BM25 retrieval scores for every sentence and row in the source documents. For example, `train.json` is originally **74.6 MB**.

We built a custom ETL pipeline in `preprocess.py` to extract only the target keys required for our generator fine-tuning and evaluation loops. 
*   **Storage Reduction:** Shrunk dataset footprint by **54%** across all splits.
*   **Performance Optimization:** Reduces memory-overhead and file I/O loading times during training runs.
*   **Hermetic Environment:** All files are stored locally to bypass online Hugging Face hub latency or availability issues.
