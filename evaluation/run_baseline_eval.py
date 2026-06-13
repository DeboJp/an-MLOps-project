import os
import sys
import json
import argparse
from tqdm import tqdm

# Configure package path imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.client import OllamaClient
from src.retriever import FinQARetriever
from src.evaluator import program_tokenization, evaluate_result

def clean_llm_response(raw_text):
    """
    Cleans up any potential formatting artifacts from the LLM.
    We expect a format like: subtract(65000, 50000), divide(#0, 50000), EOF
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.replace("```text", "").replace("```json", "").replace("```", "")
    text = text.strip()
    
    # Remove EOF because program_tokenization appends 'EOF' automatically
    text = text.replace("EOF", "")
    text = text.strip().strip(",")
    
    # Ensure there is a space after every comma to satisfy program_tokenization's split(', ')
    text = text.replace(",", ", ")
    while "  " in text:
        text = text.replace("  ", " ")
        
    return text.strip()

def format_gold_prompt(example):
    """
    Constructs the prompt using the entire table and gold sentences.
    """
    table_rows = [" | ".join(row) for row in example["table"]]
    table_str = "\n".join(table_rows)

    gold_texts = list(example["qa"]["gold_inds"].values())
    context_str = "\n".join(gold_texts)

    prompt = f"""Context:
Table:
{table_str}

Text:
{context_str}

Question:
{example["qa"]["question"]}
"""
    return prompt

def format_retrieved_prompt(example, retrieved_docs):
    """
    Constructs the prompt using only the retrieved documents/rows.
    """
    context_str = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""Context:
{context_str}

Question:
{example["qa"]["question"]}
"""
    return prompt

def format_gold_chunks_prompt(example):
    """
    Constructs the prompt using only the gold sentences and table rows
    formatted in the exact Header-Aware Column chunking template.
    """
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])
    table = example.get("table", [])
    gold_inds = example["qa"].get("gold_inds", {})

    gold_chunks = []

    def clean_txt(t):
        return t.strip()

    # Pre-text sentences
    for i, text in enumerate(pre_text):
        content = clean_txt(text)
        if not content:
            continue
        key = f"text_{i}"
        if key in gold_inds:
            gold_chunks.append(content)

    # Post-text sentences
    pre_len = len(pre_text)
    for i, text in enumerate(post_text):
        content = clean_txt(text)
        if not content:
            continue
        key = f"text_{pre_len + i}"
        if key in gold_inds:
            gold_chunks.append(content)

    # Table rows with Header-Aware chunking format
    if len(table) > 0:
        headers = table[0]
        headers_str = " | ".join(headers)
        start_idx = 1 if len(table) > 1 else 0
        
        # Row 0
        key_0 = "table_0"
        if key_0 in gold_inds:
            content_0 = f"Columns: {headers_str} -> Row: {headers_str}"
            gold_chunks.append(content_0)

        # Rest of table
        for i in range(1, len(table)):
            key = f"table_{i}"
            if key in gold_inds:
                row_str = " | ".join(table[i])
                content = f"Columns: {headers_str} -> Row: {row_str}"
                gold_chunks.append(content)

    context_str = "\n".join(gold_chunks)
    prompt = f"""Context:
{context_str}

Question:
{example["qa"]["question"]}
"""
    return prompt

def run_evaluation(dataset_path=None, limit=10, model_name="finqa-llama3.2", mode="gold", show_comparison=False):
    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "data", "test.json")
    elif not os.path.isabs(dataset_path):
        dataset_path = os.path.join(PROJECT_ROOT, dataset_path)

    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    eval_dataset = dataset[:limit]
    print(f"Running evaluation on first {len(eval_dataset)} examples in '{mode}' mode using model '{model_name}'...")

    client = OllamaClient(model=model_name)
    retriever = FinQARetriever() if mode == "retrieved" else None
    predictions = []
    
    total_tps = 0.0
    total_ttft = 0.0
    perf_count = 0

    for example in tqdm(eval_dataset):
        try:
            if mode == "retrieved":
                retrieved_docs = retriever.retrieve_context(example, example["qa"]["question"], k=5)
                prompt = format_retrieved_prompt(example, retrieved_docs)
            elif mode == "gold_chunks":
                prompt = format_gold_chunks_prompt(example)
            else:
                prompt = format_gold_prompt(example)
                
            res = client.generate(prompt)
            raw_response = res["text"]
            cleaned_response = clean_llm_response(raw_response)
            pred_tokens = program_tokenization(cleaned_response)
            
            if "tps" in res and res["tps"] > 0:
                total_tps += res["tps"]
                total_ttft += res["ttft"]
                perf_count += 1
        except Exception as e:
            tqdm.write(f"\n[Warning] Error processing example {example['id']}: {e}")
            pred_tokens = ["error", "EOF"]
            cleaned_response = f"error: {e}"

        expected_program = example["qa"]["program"]

        if show_comparison:
            tqdm.write("\n" + "="*50)
            tqdm.write(f"Question: {example['qa']['question']}")
            if mode == "retrieved" and 'retrieved_docs' in locals():
                tqdm.write("Retrieved Context:")
                for i, doc in enumerate(retrieved_docs):
                    tqdm.write(f"  [{i}] {doc.page_content} ({doc.metadata})")
            elif mode == "gold_chunks":
                tqdm.write("Gold Chunks Context:")
                tqdm.write(prompt.replace("Context:\n", "").replace(f"\n\nQuestion:\n{example['qa']['question']}\n", ""))
            
            tqdm.write("Expected (Gold) Context:")
            gold_texts = list(example["qa"]["gold_inds"].values())
            for i, gold_text in enumerate(gold_texts):
                tqdm.write(f"  [{i}] {gold_text}")
                
            tqdm.write(f"LLM Output: {cleaned_response}")
            tqdm.write(f"Expected:   {expected_program}")
            tqdm.write("="*50 + "\n")

        predictions.append({
            "id": example["id"],
            "predicted": pred_tokens
        })

    # Save predictions to file in the root directory
    pred_path = os.path.join(PROJECT_ROOT, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Saved predictions to: {pred_path}")

    # Map model to serving location
    model_lower = model_name.lower()
    if "sagemaker" in model_lower:
        serving_loc = "AWS SageMaker Real-Time Endpoint (InService)"
    elif model_name == "finqa-llama3.2-tuned":
        serving_loc = "Local MLX-LM Server (Port 11435)"
    else:
        serving_loc = "Local Ollama Server (Port 11434)"

    exe_correct, prog_correct, total = evaluate_result(pred_path, dataset_path)

    exe_acc = float(exe_correct) / total if total > 0 else 0.0
    prog_acc = float(prog_correct) / total if total > 0 else 0.0

    avg_tps = total_tps / perf_count if perf_count > 0 else 0.0
    avg_ttft = total_ttft / perf_count if perf_count > 0 else 0.0

    # Print clean and simple evaluation summary
    print("\n--- Evaluation Summary ---")
    print(f"Model:            {model_name}")
    print(f"Served At:        {serving_loc}")
    print(f"Mode:             {mode}")
    print(f"All:              {total}")
    print(f"Exe acc:          {exe_acc:.4f} ({exe_correct}/{total})")
    print(f"Prog acc:         {prog_acc:.4f} ({prog_correct}/{total})")
    if avg_tps > 0:
        print(f"Avg TPS:          {avg_tps:.2f}")
        print(f"Avg TTFT:         {avg_ttft:.3f} s")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinQA MLOps Evaluation Runner")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset JSON file")
    parser.add_argument("--limit", type=int, default=3, help="Number of examples to evaluate")
    parser.add_argument("--model", type=str, default="finqa-llama3.2", help="Ollama model name")
    parser.add_argument("--mode", type=str, choices=["gold", "retrieved", "gold_chunks"], default="gold", help="Evaluation mode")
    parser.add_argument("--show-comparison", action="store_true", help="Print comparison between LLM output and expected ground truth")
    args = parser.parse_args()

    run_evaluation(args.dataset, limit=args.limit, model_name=args.model, mode=args.mode, show_comparison=args.show_comparison)
