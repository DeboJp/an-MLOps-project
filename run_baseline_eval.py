import os
import sys
import json
import argparse
from tqdm import tqdm

# Import the Ollama client
from ollama_client import OllamaClient
# Import the FinQARetriever
from rag.retriever import FinQARetriever

# Add FinQA evaluate directory to sys.path so we can import evaluate modules
sys.path.append(os.path.join(os.path.dirname(__file__), "FinQA-main", "code", "evaluate"))
from evaluate import program_tokenization, evaluate_result

def clean_llm_response(raw_text):
    """
    Cleans up any potential formatting artifacts from the LLM.
    We expect a format like: subtract(65000, 50000), divide(#0, 50000), EOF
    """
    text = raw_text.strip()
    # Remove markdown code blocks if any
    if text.startswith("```"):
        text = text.replace("```text", "").replace("```json", "").replace("```", "")
    text = text.strip()
    
    # Remove EOF because program_tokenization appends 'EOF' automatically
    text = text.replace("EOF", "")
    
    # Strip any trailing commas or whitespace after removing EOF
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

def run_evaluation(dataset_path, limit=10, model_name="finqa-llama3.2", mode="gold", show_comparison=False):
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    eval_dataset = dataset[:limit]
    print(f"Running evaluation on first {len(eval_dataset)} examples in '{mode}' mode using model '{model_name}'...")

    client = OllamaClient(model=model_name)
    retriever = FinQARetriever() if mode == "retrieved" else None
    predictions = []

    for example in tqdm(eval_dataset):
        try:
            if mode == "retrieved":
                retrieved_docs = retriever.retrieve_context(example, example["qa"]["question"], k=5)
                prompt = format_retrieved_prompt(example, retrieved_docs)
            else:
                prompt = format_gold_prompt(example)
                
            res = client.generate(prompt)
            raw_response = res["text"]
            cleaned_response = clean_llm_response(raw_response)
            
            # Tokenize the prediction using the official tokenization logic
            pred_tokens = program_tokenization(cleaned_response)
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

    # Save predictions to file
    pred_path = "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Saved predictions to: {pred_path}")

    print("\n--- Running Evaluation Metrics ---")
    evaluate_result(pred_path, dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinQA MLOps Evaluation Runner")
    parser.add_argument("--dataset", type=str, default="data/test.json", help="Path to the dataset JSON file")
    parser.add_argument("--limit", type=int, default=3, help="Number of examples to evaluate")
    parser.add_argument("--model", type=str, default="finqa-llama3.2", help="Ollama model name")
    parser.add_argument("--mode", type=str, choices=["gold", "retrieved"], default="gold", help="Evaluation mode")
    parser.add_argument("--show-comparison", action="store_true", help="Print comparison between LLM output and expected ground truth")
    args = parser.parse_args()

    run_evaluation(args.dataset, limit=args.limit, model_name=args.model, mode=args.mode, show_comparison=args.show_comparison)

