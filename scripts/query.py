"""
Interactive inference script to run the hybrid RAG pipeline on individual FinQA dataset examples.
Demonstrates the end-to-end flow of dynamic retrieval, prompt formatting, and Ollama model query.
"""

import argparse
import json
import os
import sys

# Configure package path imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.retriever import FinQARetriever
from src.client import OllamaClient
from evaluation.run_baseline_eval import clean_llm_response, format_retrieved_prompt

def query_single_example(example_index=0, dataset_path=None, model_name="finqa-llama3.2"):
    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "data", "test.json")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if example_index >= len(dataset) or example_index < 0:
        print(f"Error: Index {example_index} is out of bounds (dataset size: {len(dataset)})")
        return

    example = dataset[example_index]
    question = example["qa"]["question"]

    print("\n" + "="*80)
    print(f"Executing interactive RAG query for Example #{example_index}")
    print(f"Question: {question}")
    print("="*80 + "\n")

    # Step 1: Retrieval
    print("[1/3] Initializing Hybrid Retriever and building transient index...")
    retriever = FinQARetriever()
    retrieved_docs = retriever.retrieve_context(example, question, k=5)
    
    print("\n--- Retrieved Context Docs (Top 5) ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i}] {doc.page_content} ({doc.metadata})")
    print("-" * 38 + "\n")

    # Step 2: Prompt Formatting
    print("[2/3] Preparing prompt for Ollama/MLX...")
    prompt = format_retrieved_prompt(example, retrieved_docs)

    # Step 3: Inference
    print(f"[3/3] Querying model '{model_name}'...")
    client = OllamaClient(model=model_name)
    res = client.generate(prompt)
    raw_response = res["text"]
    cleaned_response = clean_llm_response(raw_response)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"LLM Raw Output:     {raw_response.strip()}")
    print(f"LLM Cleaned Program: {cleaned_response}")
    print(f"Expected Program:    {example['qa']['program']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query FinQA pipeline interactively")
    parser.add_argument("--index", type=int, default=0, help="Index of the dataset example to run")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset json")
    parser.add_argument("--model", type=str, default="finqa-llama3.2", help="Ollama or MLX model identifier")
    args = parser.parse_args()

    query_single_example(example_index=args.index, dataset_path=args.dataset, model_name=args.model)
