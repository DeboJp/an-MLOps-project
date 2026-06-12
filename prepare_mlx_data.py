"""
Script to prepare MLX-compatible fine-tuning datasets from FinQA splits.
Utilizes a hybrid Gold + Noisy Padding approach:
1. Always includes the exact ground-truth (gold) text sentences and table rows.
2. Formats table rows using the Header-Aware Column chunking template.
3. Fills up the context budget (k=5) with randomly sampled non-gold distractors from the same page.
4. Shuffles the combined chunks to prevent position bias.
"""

import os
import json
import argparse
import random
from tqdm import tqdm

SYSTEM_PROMPT = """You are a financial calculator assistant. Your task is to output a mathematical program using the FinQA Domain-Specific Language (DSL) to answer the user's question based on the provided text and table.

Allowed operations:
- add(arg1, arg2)
- subtract(arg1, arg2)
- multiply(arg1, arg2)
- divide(arg1, arg2)
- exp(arg1, arg2)
- greater(arg1, arg2)
- table_max(row_name)
- table_min(row_name)
- table_sum(row_name)
- table_average(row_name)

Refer to the results of previous steps using #0, #1, etc.
Format your output exactly as a comma-separated sequence of operations, ending with EOF. Do not include any other explanations, markdown formatting, or conversational filler.

Example 1:
Context:
Table:
| Year | Revenue |
| 2020 | 50000 |
| 2021 | 65000 |
Question: What was the rate of increase of revenue from 2020 to 2021?
Output: subtract(65000, 50000), divide(#0, 50000), EOF

CRITICAL: You must extract the exact literal numbers from the table. Do not use placeholders. Each problem will be unique, and require basic arithmetic decision making. Count digits carefully."""


def get_gold_and_noisy_chunks(example, k=5):
    """
    Retrieves the exact raw gold contexts, formats them in the retriever layout,
    pads the rest up to k chunks with non-gold distractors from the same page,
    and shuffles the combined output.
    """
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])
    table = example.get("table", [])
    gold_inds = example["qa"].get("gold_inds", {})

    gold_chunks = []
    non_gold_chunks = []

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
        else:
            non_gold_chunks.append(content)

    # Post-text sentences
    pre_len = len(pre_text)
    for i, text in enumerate(post_text):
        content = clean_txt(text)
        if not content:
            continue
        key = f"text_{pre_len + i}"
        if key in gold_inds:
            gold_chunks.append(content)
        else:
            non_gold_chunks.append(content)

    # Table rows with Header-Aware chunking format
    if len(table) > 0:
        headers = table[0]
        headers_str = " | ".join(headers)
        start_idx = 1 if len(table) > 1 else 0
        
        # If the table has only 1 row
        if start_idx == 0:
            content = f"Columns: {headers_str} -> Row: {headers_str}"
            key = "table_0"
            if key in gold_inds:
                gold_chunks.append(content)
            else:
                non_gold_chunks.append(content)
        else:
            # Row 0 (headers) can occasionally be in gold_inds
            key_0 = "table_0"
            content_0 = f"Columns: {headers_str} -> Row: {headers_str}"
            if key_0 in gold_inds:
                gold_chunks.append(content_0)
            else:
                non_gold_chunks.append(content_0)

            # Rest of the data rows
            for i in range(1, len(table)):
                row = table[i]
                row_str = " | ".join(row)
                if not row_str.strip():
                    continue
                content = f"Columns: {headers_str} -> Row: {row_str}"
                key = f"table_{i}"
                if key in gold_inds:
                    gold_chunks.append(content)
                else:
                    non_gold_chunks.append(content)

    # Fill remaining slots from non-gold chunks
    num_to_sample = max(0, k - len(gold_chunks))
    if len(non_gold_chunks) > num_to_sample:
        sampled_noise = random.sample(non_gold_chunks, num_to_sample)
    else:
        sampled_noise = non_gold_chunks

    # Combine and shuffle to prevent position bias
    combined_chunks = gold_chunks + sampled_noise
    random.shuffle(combined_chunks)
    return combined_chunks


def format_context_prompt(question, chunks):
    """
    Formats the context chunks and question into the prompt template.
    """
    context_str = "\n".join(chunks)
    prompt = f"""Context:
{context_str}

Question:
{question}
"""
    return prompt


def prepare_split(dataset_path, output_path, limit=None):
    print(f"Loading raw dataset from {dataset_path}...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Set random seed for reproducible sampling and shuffling
    random.seed(42)

    if limit and limit < len(data):
        data = random.sample(data, limit)

    print(f"Generating MLX chat dataset at {output_path} (Gold + Noisy Padding)...")
    
    with open(output_path, "w") as out_file:
        for idx, example in enumerate(tqdm(data)):
            question = example["qa"]["question"]
            program = example["qa"]["program"]
            
            # Retrieve aligned chunks using Gold + Noisy Padding
            chunks = get_gold_and_noisy_chunks(example, k=5)
            user_content = format_context_prompt(question, chunks)
            
            # Format in Llama-3.2-Instruct chat structure
            chat_format = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"{program}, EOF"}
                ]
            }
            
            out_file.write(json.dumps(chat_format) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare MLX datasets with gold + noisy padding")
    parser.add_argument("--train-limit", type=int, default=500, help="Number of train examples to process")
    parser.add_argument("--val-limit", type=int, default=50, help="Number of validation examples to process")
    parser.add_argument("--test-limit", type=int, default=100, help="Number of test examples to process")
    args = parser.parse_args()

    os.makedirs("mlx_data", exist_ok=True)

    # Process splits
    prepare_split("data/train.json", "mlx_data/train.jsonl", limit=args.train_limit)
    prepare_split("data/val.json", "mlx_data/valid.jsonl", limit=args.val_limit)
    prepare_split("data/test.json", "mlx_data/test.jsonl", limit=args.test_limit)
    
    print("\nDataset preparation complete! Files written to mlx_data/")


if __name__ == "__main__":
    main()
