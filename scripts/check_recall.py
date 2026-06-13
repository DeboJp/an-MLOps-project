import json
import sys
import os

# Configure package path imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.retriever import FinQARetriever

def get_gold_sentences_and_rows(example):
    """
    Extracts the actual content of the gold sentences and rows as strings.
    """
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])
    table = example.get("table", [])
    gold_inds = example["qa"].get("gold_inds", {})

    gold_contents = []

    # Pre-text sentences
    for i, text in enumerate(pre_text):
        key = f"text_{i}"
        if key in gold_inds:
            gold_contents.append(text.strip())

    # Post-text sentences
    pre_len = len(pre_text)
    for i, text in enumerate(post_text):
        key = f"text_{pre_len + i}"
        if key in gold_inds:
            gold_contents.append(text.strip())

    # Table rows
    if len(table) > 0:
        headers = table[0]
        headers_str = " | ".join(headers)
        start_idx = 1 if len(table) > 1 else 0
        
        # Row 0
        key_0 = "table_0"
        if key_0 in gold_inds:
            if start_idx == 0:
                gold_contents.append(f"Columns: {headers_str} -> Row: {headers_str}")
            else:
                gold_contents.append(f"Columns: {headers_str} -> Row: {headers_str}")

        # Rest of table
        for i in range(1, len(table)):
            key = f"table_{i}"
            if key in gold_inds:
                row_str = " | ".join(table[i])
                gold_contents.append(f"Columns: {headers_str} -> Row: {row_str}")

    return gold_contents

def analyze_recall(limit=20):
    dataset_path = os.path.join(PROJECT_ROOT, "data", "test.json")
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    eval_dataset = dataset[:limit]
    print(f"Analyzing retriever recall on the first {limit} examples...")

    retriever = FinQARetriever()
    
    total_gold_chunks = 0
    retrieved_gold_chunks = 0
    fully_recalled_examples = 0

    for example in eval_dataset:
        question = example["qa"]["question"]
        gold_chunks = get_gold_sentences_and_rows(example)
        
        # Retrieve the context
        retrieved_docs = retriever.retrieve_context(example, question, k=5)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        # Check overlap
        matched = 0
        for gold in gold_chunks:
            # We check for substring matches or exact matches
            if any(gold in ret or ret in gold for ret in retrieved_texts):
                matched += 1
        
        total_gold_chunks += len(gold_chunks)
        retrieved_gold_chunks += matched
        
        if matched == len(gold_chunks):
            fully_recalled_examples += 1

    chunk_recall = (retrieved_gold_chunks / total_gold_chunks) * 100 if total_gold_chunks > 0 else 0
    full_recall_rate = (fully_recalled_examples / len(eval_dataset)) * 100 if len(eval_dataset) > 0 else 0

    print("\n--- Retriever Diagnostics Summary ---")
    print(f"Evaluated Examples: {len(eval_dataset)}")
    print(f"Total Gold Chunks Needed: {total_gold_chunks}")
    print(f"Gold Chunks Successfully Retrieved: {retrieved_gold_chunks}")
    print(f"Chunk-Level Recall Accuracy: {chunk_recall:.2f}%")
    print(f"Perfect Recall Rate (Examples where 100% of gold chunks were retrieved): {full_recall_rate:.2f}%")
    print("--------------------------------------")

if __name__ == "__main__":
    analyze_recall()
