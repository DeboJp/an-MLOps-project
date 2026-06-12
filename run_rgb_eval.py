"""
Evaluation driver script for running RAG RGB testbeds on FinQA models.

Measures model performance across the 4 RGB failure modes:
1. Noise Robustness (high noise k=10)
2. Negative Rejection (evaluating refusal output 'reject, EOF')
3. Information Integration (len(gold_inds) >= 2)
4. Counterfactual Robustness (modifying context numbers and evaluating formula alignment)
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Import the OllamaClient and base SYSTEM_PROMPT
from ollama_client import OllamaClient, SYSTEM_PROMPT

# Add FinQA evaluate directory to sys.path to import official tokenization and equality checks
sys.path.append(os.path.join(os.path.dirname(__file__), "FinQA-main", "code", "evaluate"))
from evaluate import program_tokenization, eval_program, equal_program

# Append the refusal instruction to the system prompt for Negative Rejection testbeds
# Explicitly define how the model should behave under missing contexts.
REJECTION_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\nIf the provided context does not contain sufficient information to answer the question, output reject, EOF."

def clean_llm_response(raw_text):
    """
    Cleans up the raw output text from the LLM, aligning it with the expected DSL tokenization.
    Example: 'subtract(65000, 50000), EOF' -> 'subtract(65000, 50000)'
    """
    text = raw_text.strip()
    # Remove markdown formatting
    if text.startswith("```"):
        text = text.replace("```text", "").replace("```json", "").replace("```", "")
    text = text.strip()
    
    # Remove EOF
    text = text.replace("EOF", "")
    text = text.strip().strip(",")
    
    # Ensure standard comma-spacing format for evaluate split(', ')
    text = text.replace(",", ", ")
    while "  " in text:
        text = text.replace("  ", " ")
        
    return text.strip()

def evaluate_testbed(testbed_name, dataset_path, client, limit=50, show_comparison=False):
    """
    Runs evaluation for a specific testbed dataset, calculating execution accuracy,
    program accuracy, and format structure errors.
    """
    print(f"\nEvaluating testbed: '{testbed_name}'...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Run generate_rgb_testbeds.py first.")
        return None
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    eval_dataset = dataset[:limit]
    print(f"Running on first {len(eval_dataset)} examples...")
    
    exe_correct = 0
    prog_correct = 0
    structure_errors = 0
    total = len(eval_dataset)
    
    # For Negative Rejection, we override the system prompt to instruct the model to reject
    sys_prompt = REJECTION_SYSTEM_PROMPT if testbed_name == "negative_rejection" else SYSTEM_PROMPT
    
    for idx, example in enumerate(tqdm(eval_dataset)):
        question = example["qa"]["question"]
        gold_program = example["qa"]["program"]
        gold_exe_ans = example["qa"]["exe_ans"]
        table = example["table"]
        eval_context = example["qa"]["eval_context"]
        
        # Format the user prompt
        user_prompt = f"Context:\n{eval_context}\n\nQuestion:\n{question}\n"
        
        try:
            res = client.generate(user_prompt, system_prompt=sys_prompt)
            raw_response = res["text"]
            cleaned_response = clean_llm_response(raw_response)
            
            # Tokenize using the official tokenization logic
            pred_tokens = program_tokenization(cleaned_response)
        except Exception as e:
            tqdm.write(f"Error executing prompt for example {example['id']}: {e}")
            pred_tokens = ["error", "EOF"]
            cleaned_response = f"error: {e}"
            
        # 1. Evaluate Structure & Program correctness
        gold_tokens = program_tokenization(gold_program)
        
        # Check if model has a structure error (first token or structure is incorrect)
        # Note: If the testbed expects rejection, 'reject' is a valid single token
        is_structure_error = False
        if len(pred_tokens) > 0 and pred_tokens[0] not in ["reject", "error"]:
            # Standard structural checks from evaluate.py
            for t_idx, token in enumerate(pred_tokens[:-1]):
                if t_idx % 4 == 0:
                    if token.strip("(") not in ["add", "subtract", "multiply", "divide", "exp", "greater", 
                                               "table_max", "table_min", "table_sum", "table_average"]:
                        is_structure_error = True
                        break
                        
        if is_structure_error or "error" in pred_tokens:
            structure_errors += 1
            
        # 2. Check program accuracy
        is_prog_match = equal_program(gold_tokens, pred_tokens)
        if is_prog_match:
            prog_correct += 1
            
        # 3. Check execution accuracy
        if testbed_name == "negative_rejection":
            # For Negative Rejection, correct answer is when output matches 'reject'
            if len(pred_tokens) > 0 and pred_tokens[0] == "reject":
                exe_correct += 1
        else:
            # Run official eval_program to compute target value
            try:
                invalid_flag, pred_exe_val = eval_program(pred_tokens, table)
                if invalid_flag == 0 and pred_exe_val == gold_exe_ans:
                    exe_correct += 1
            except Exception:
                pass
                
        if show_comparison:
            tqdm.write("\n" + "="*60)
            tqdm.write(f"ID:       {example['id']}")
            tqdm.write(f"Question: {question}")
            tqdm.write(f"LLM Out:  {cleaned_response}")
            tqdm.write(f"Expected: {gold_program}")
            if testbed_name != "negative_rejection":
                tqdm.write(f"LLM Val:  {locals().get('pred_exe_val', 'n/a')}")
                tqdm.write(f"Gold Val: {gold_exe_ans}")
            tqdm.write("="*60 + "\n")
            
    # Compile metrics
    exe_acc = exe_correct / total if total > 0 else 0
    prog_acc = prog_correct / total if total > 0 else 0
    struct_err_pct = structure_errors / total if total > 0 else 0
    
    return {
        "total": total,
        "exe_acc": exe_acc,
        "prog_acc": prog_acc,
        "struct_err": struct_err_pct
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate FinQA models on RGB Testbeds")
    parser.add_argument("--model", type=str, default="finqa-llama3.2", help="Ollama or MLX-LM model name")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples per testbed to evaluate")
    parser.add_argument("--testbed", type=str, choices=["all", "noise_robustness", "negative_rejection", 
                                                        "information_integration", "counterfactual_robustness"], 
                        default="all", help="Specific testbed to run")
    parser.add_argument("--show-comparison", action="store_true", help="Print model answers during run")
    args = parser.parse_args()

    # Load client (will dynamically route based on port 11434 / 11435)
    client = OllamaClient(model=args.model)

    testbeds = {
        "noise_robustness": "data/rgb_eval/noise_robustness.json",
        "negative_rejection": "data/rgb_eval/negative_rejection.json",
        "information_integration": "data/rgb_eval/information_integration.json",
        "counterfactual_robustness": "data/rgb_eval/counterfactual_robustness.json"
    }

    if args.testbed != "all":
        testbeds = {args.testbed: testbeds[args.testbed]}

    results = {}

    for name, path in testbeds.items():
        res = evaluate_testbed(name, path, client, limit=args.limit, show_comparison=args.show_comparison)
        if res:
            results[name] = res

    # Print summary table
    print("\n" + "="*80)
    print(f" RAG RGB Testbed Evaluation Summary for Model: {args.model}")
    print("="*80)
    print(f"{'Testbed Name':<30} | {'Count':<5} | {'Prog Acc':<10} | {'Exe Acc':<10} | {'Struct Err':<10}")
    print("-"*80)
    for name, metrics in results.items():
        # Label adjustment for negative rejection (since execution acc is rejection correctness)
        exe_label = f"{metrics['exe_acc']*100:.1f}%"
        prog_label = f"{metrics['prog_acc']*100:.1f}%"
        
        if name == "negative_rejection":
            exe_label = f"{metrics['exe_acc']*100:.1f}% (Rej)"
            prog_label = "N/A"
            
        print(f"{name:<30} | {metrics['total']:<5} | {prog_label:<10} | {exe_label:<10} | {metrics['struct_err']*100:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
