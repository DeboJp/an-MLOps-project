"""
Script to generate the 4 RGB evaluation testbeds (Noise Robustness, Negative Rejection,
Information Integration, and Counterfactual Robustness) from data/test.json.

Defaults to selecting the first 50 matching examples for reproducibility, 
with a CLI flag to select random samples instead.
"""

import os
import sys
import json
import argparse
import random
import re

# Add project root path to allow modular imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.evaluator import program_tokenization, eval_program

def get_page_chunks(example):
    """
    Helper function to extract all possible chunks (text sentences and table rows)
    from a single example page.
    """
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])
    table = example.get("table", [])
    
    chunks = []
    
    # 1. Add pre-text sentences
    for t in pre_text:
        content = t.strip()
        if content:
            chunks.append(content)
            
    # 2. Add post-text sentences
    for t in post_text:
        content = t.strip()
        if content:
            chunks.append(content)
            
    # 3. Add table rows with Header-Aware Column chunking
    if len(table) > 0:
        headers = table[0]
        headers_str = " | ".join(headers)
        start_idx = 1 if len(table) > 1 else 0
        for i in range(start_idx, len(table)):
            row = table[i]
            row_str = " | ".join(row)
            if row_str.strip():
                chunks.append(f"Columns: {headers_str} -> Row: {row_str}")
                
    return chunks


def get_gold_chunks_and_distractors(example):
    """
    Decodes the gold_inds to find actual gold chunks, and separate them from
    distractors (non-gold chunks) from the same page.
    """
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])
    table = example.get("table", [])
    gold_inds = example["qa"].get("gold_inds", {})
    
    gold_chunks = []
    distractor_chunks = []
    
    # Decode pre-text
    for i, text in enumerate(pre_text):
        content = text.strip()
        if not content:
            continue
        key = f"text_{i}"
        if key in gold_inds:
            gold_chunks.append(content)
        else:
            distractor_chunks.append(content)
            
    # Decode post-text
    pre_len = len(pre_text)
    for i, text in enumerate(post_text):
        content = text.strip()
        if not content:
            continue
        key = f"text_{pre_len + i}"
        if key in gold_inds:
            gold_chunks.append(content)
        else:
            distractor_chunks.append(content)
            
    # Decode table
    if len(table) > 0:
        headers = table[0]
        headers_str = " | ".join(headers)
        start_idx = 1 if len(table) > 1 else 0
        
        # Row 0 (headers)
        key_0 = "table_0"
        content_0 = f"Columns: {headers_str} -> Row: {headers_str}"
        if key_0 in gold_inds:
            gold_chunks.append(content_0)
        else:
            distractor_chunks.append(content_0)
            
        # Data rows
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
                distractor_chunks.append(content)
                
    return gold_chunks, distractor_chunks


def generate_noise_robustness(data, limit=50, select_random=False):
    """
    Noise Robustness Testbed:
    Selects examples, constructs prompt context with ALL gold chunks + 8 distractors (k=10 chunks total).
    The query asks the original question. Expected output is the original program.
    """
    selected = []
    source_data = list(data)
    if select_random:
        random.shuffle(source_data)
        
    for example in source_data:
        gold, distractors = get_gold_chunks_and_distractors(example)
        if not gold:
            continue  # Need gold chunks to test robustness
            
        # Sample distractors to fill up to k=10 total chunks
        needed_noise = max(0, 10 - len(gold))
        if len(distractors) >= needed_noise:
            sampled_noise = random.sample(distractors, needed_noise)
        else:
            sampled_noise = distractors
            
        combined = gold + sampled_noise
        random.shuffle(combined)
        
        # Build prompt's context part
        context_str = "\n".join(combined)
        
        # Create eval object
        new_example = {
            "id": example["id"],
            "pre_text": example["pre_text"],
            "post_text": example["post_text"],
            "table": example["table"],
            "qa": {
                "question": example["qa"]["question"],
                "program": example["qa"]["program"],
                "exe_ans": example["qa"]["exe_ans"],
                "gold_inds": example["qa"]["gold_inds"],
                "eval_context": context_str
            }
        }
        selected.append(new_example)
        if len(selected) == limit:
            break
            
    return selected


def generate_negative_rejection(data, limit=50, select_random=False):
    """
    Negative Rejection Testbed:
    Selects examples, constructs prompt context with ONLY distractor chunks (5 chunks, 0 gold chunks).
    The model is queried with the original question, but because context lacks gold information,
    it must output "reject, EOF".
    """
    selected = []
    source_data = list(data)
    if select_random:
        random.shuffle(source_data)
        
    for example in source_data:
        gold, distractors = get_gold_chunks_and_distractors(example)
        # We need to make sure we have enough distractors (at least 3) and that gold is not empty
        # (which proves the question actually had an answer originally but now doesn't).
        if not gold or len(distractors) < 3:
            continue
            
        # Sample 5 distractors (or all if less than 5)
        k_noise = min(5, len(distractors))
        sampled_noise = random.sample(distractors, k_noise)
        random.shuffle(sampled_noise)
        
        context_str = "\n".join(sampled_noise)
        
        new_example = {
            "id": example["id"],
            "pre_text": example["pre_text"],
            "post_text": example["post_text"],
            "table": example["table"],
            "qa": {
                "question": example["qa"]["question"],
                "program": "reject",  # Expected refusal output
                "exe_ans": "n/a",      # Rejection execution answer
                "gold_inds": example["qa"]["gold_inds"],
                "eval_context": context_str
            }
        }
        selected.append(new_example)
        if len(selected) == limit:
            break
            
    return selected


def generate_information_integration(data, limit=50, select_random=False):
    """
    Information Integration Testbed:
    Selects examples where len(gold_inds) >= 2 (requires multi-hop reasoning).
    Context is gold + 3 distractors (k=5 chunks total).
    """
    selected = []
    source_data = list(data)
    if select_random:
        random.shuffle(source_data)
        
    for example in source_data:
        gold, distractors = get_gold_chunks_and_distractors(example)
        if len(gold) < 2:
            continue  # Must require integrating at least 2 distinct chunks
            
        # Sample distractors to fill up to k=5 total chunks
        needed_noise = max(0, 5 - len(gold))
        if len(distractors) >= needed_noise:
            sampled_noise = random.sample(distractors, needed_noise)
        else:
            sampled_noise = distractors
            
        combined = gold + sampled_noise
        random.shuffle(combined)
        
        context_str = "\n".join(combined)
        
        new_example = {
            "id": example["id"],
            "pre_text": example["pre_text"],
            "post_text": example["post_text"],
            "table": example["table"],
            "qa": {
                "question": example["qa"]["question"],
                "program": example["qa"]["program"],
                "exe_ans": example["qa"]["exe_ans"],
                "gold_inds": example["qa"]["gold_inds"],
                "eval_context": context_str
            }
        }
        selected.append(new_example)
        if len(selected) == limit:
            break
            
    return selected


def generate_counterfactual_robustness(data, limit=50, select_random=False):
    """
    Counterfactual Robustness Testbed:
    Parses numbers in the program that come from the context. Replaces them in context, gold_inds, 
    and expected program with random counterfactual values of similar shape/magnitude. 
    Verifies that the modified program executes successfully on the modified table.
    """
    selected = []
    source_data = list(data)
    if select_random:
        random.shuffle(source_data)
        
    for example in source_data:
        gold, distractors = get_gold_chunks_and_distractors(example)
        if not gold:
            continue
            
        program = example["qa"]["program"]
        
        # 1. Identify numbers in program (ignore references like #0, #1, and common small constants)
        raw_numbers = re.findall(r'(?<!#)\b\d+(?:\.\d+)?%?\b', program)
        raw_numbers = list(set(raw_numbers))
        
        modifiable = []
        for num_str in raw_numbers:
            clean_num = num_str.replace("%", "").strip()
            if not clean_num or clean_num in ["1", "2", "3", "4", "5", "6", "10", "12", "100", "1000"]:
                continue
                
            # Verify if this number exists in the gold context
            in_gold = False
            for chunk in gold:
                if clean_num in chunk:
                    in_gold = True
                    break
            if in_gold:
                modifiable.append((num_str, clean_num))
                
        if not modifiable:
            continue
            
        # 2. Map old values to counterfactual ones
        value_mapping = {}
        for num_str, clean_num in modifiable:
            has_pct = "%" in num_str
            try:
                if "." in clean_num:
                    val = float(clean_num)
                    # Add random offset between 10.0 and 50.0
                    new_val = round(val + random.uniform(10.0, 50.0), 2)
                    new_clean = f"{new_val}"
                else:
                    val = int(clean_num)
                    # If it's a year, shift it by 50-80 years
                    if 1900 <= val <= 2100:
                        new_val = val + random.randint(50, 80)
                    else:
                        # Otherwise shift by a random integer offset
                        new_val = val + random.randint(100, 999)
                    new_clean = f"{new_val}"
            except ValueError:
                new_clean = clean_num
                
            new_num_str = new_clean + "%" if has_pct else new_clean
            value_mapping[num_str] = new_num_str
            value_mapping[clean_num] = new_clean

        # 3. Create modified copies of text, tables, and program
        mod_pre_text = list(example.get("pre_text", []))
        mod_post_text = list(example.get("post_text", []))
        mod_table = [list(row) for row in example.get("table", [])]
        mod_gold_inds = dict(example["qa"].get("gold_inds", {}))
        mod_program = program
        
        # Replace values
        for old_val, new_val in value_mapping.items():
            mod_pre_text = [t.replace(old_val, new_val) for t in mod_pre_text]
            mod_post_text = [t.replace(old_val, new_val) for t in mod_post_text]
            mod_gold_inds = {k: v.replace(old_val, new_val) for k, v in mod_gold_inds.items()}
            mod_program = mod_program.replace(old_val, new_val)
            
            for r_idx in range(len(mod_table)):
                for c_idx in range(len(mod_table[r_idx])):
                    mod_table[r_idx][c_idx] = mod_table[r_idx][c_idx].replace(old_val, new_val)

        # 4. Validate and Execute counterfactual program on counterfactual table
        try:
            pred_tokens = program_tokenization(mod_program)
            invalid_flag, exe_res = eval_program(pred_tokens, mod_table)
            
            # Skip if calculation fails or divides by zero, or returns invalid flag
            if invalid_flag != 0 or exe_res == "n/a" or exe_res is None:
                continue
        except Exception:
            continue
            
        # 5. Build prompt's context using the modified elements
        # Regenerate chunks using modified data
        mod_example_subset = {
            "pre_text": mod_pre_text,
            "post_text": mod_post_text,
            "table": mod_table,
            "qa": {"gold_inds": mod_gold_inds}
        }
        
        mod_gold, mod_distractors = get_gold_chunks_and_distractors(mod_example_subset)
        needed_noise = max(0, 5 - len(mod_gold))
        if len(mod_distractors) >= needed_noise:
            sampled_noise = random.sample(mod_distractors, needed_noise)
        else:
            sampled_noise = mod_distractors
            
        combined = mod_gold + sampled_noise
        random.shuffle(combined)
        context_str = "\n".join(combined)
        
        new_example = {
            "id": example["id"],
            "pre_text": mod_pre_text,
            "post_text": mod_post_text,
            "table": mod_table,
            "qa": {
                "question": example["qa"]["question"],
                "program": mod_program,
                "exe_ans": exe_res,
                "gold_inds": mod_gold_inds,
                "eval_context": context_str
            }
        }
        selected.append(new_example)
        if len(selected) == limit:
            break
            
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate RGB Evaluation Testbeds for FinQA")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples per testbed")
    parser.add_argument("--random", action="store_true", help="Sample examples randomly instead of using the first matching N")
    args = parser.parse_args()

    # Set seed for repeatability
    random.seed(42)

    test_path = os.path.join(PROJECT_ROOT, "data", "test.json")
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found. Run preprocess.py first.")
        sys.exit(1)

    print(f"Loading test split from {test_path}...")
    with open(test_path, "r") as f:
        data = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "data", "rgb_eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating 4 RGB testbeds (limit={args.limit}, random={args.random}):")

    # 1. Noise Robustness
    print("- Creating Noise Robustness testbed...")
    noise_data = generate_noise_robustness(data, limit=args.limit, select_random=args.random)
    with open(os.path.join(output_dir, "noise_robustness.json"), "w") as f:
        json.dump(noise_data, f, indent=4)
    print(f"  Saved {len(noise_data)} examples.")

    # 2. Negative Rejection
    print("- Creating Negative Rejection testbed...")
    rejection_data = generate_negative_rejection(data, limit=args.limit, select_random=args.random)
    with open(os.path.join(output_dir, "negative_rejection.json"), "w") as f:
        json.dump(rejection_data, f, indent=4)
    print(f"  Saved {len(rejection_data)} examples.")

    # 3. Information Integration
    print("- Creating Information Integration testbed...")
    integration_data = generate_information_integration(data, limit=args.limit, select_random=args.random)
    with open(os.path.join(output_dir, "information_integration.json"), "w") as f:
        json.dump(integration_data, f, indent=4)
    print(f"  Saved {len(integration_data)} examples.")

    # 4. Counterfactual Robustness
    print("- Creating Counterfactual Robustness testbed...")
    counterfactual_data = generate_counterfactual_robustness(data, limit=args.limit, select_random=args.random)
    with open(os.path.join(output_dir, "counterfactual_robustness.json"), "w") as f:
        json.dump(counterfactual_data, f, indent=4)
    print(f"  Saved {len(counterfactual_data)} examples.")

    print(f"\nSuccess! All 4 RGB testbeds generated under {output_dir}/")


if __name__ == "__main__":
    main()
