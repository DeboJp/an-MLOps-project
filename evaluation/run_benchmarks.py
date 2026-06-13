#!/usr/bin/env python3
"""
FinQA Production Benchmarking Harness
Author: Debojyoti Paul
Date: 2026-06-12

Automates downstream evaluations for base and fine-tuned FinQA models.
Includes connection check guards, sweeps standard modes (gold_chunks, retrieved),
and runs the 4 RAG RGB testbeds. Outputs results to console, JSON, and Markdown.
"""

import os
import sys
import json
import time
import socket
import argparse
from datetime import datetime

# Configure path imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Local module package imports
from src.client import OllamaClient, SYSTEM_PROMPT
from src.retriever import FinQARetriever
from src.evaluator import program_tokenization, eval_program, equal_program
from evaluation.run_baseline_eval import (
    clean_llm_response,
    format_gold_prompt,
    format_retrieved_prompt,
    format_gold_chunks_prompt
)

def is_port_open(port):
    """
    Checks if a local port is listening. Safe socket connectivity check.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def run_benchmark_set(dataset_path, limit, client, set_name):
    """
    Unified benchmarking executor. Computes Execution Accuracy, Program Accuracy,
    Structure Error rate, and performance metrics (TPS, TTFT) across a dataset.
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    eval_dataset = dataset[:limit]

    retriever = FinQARetriever() if set_name == "retrieved" else None
    
    # Set refusal instruction for negative rejection testbeds
    sys_prompt = SYSTEM_PROMPT
    if set_name == "negative_rejection":
        sys_prompt = SYSTEM_PROMPT + "\n\nIf the provided context does not contain sufficient information to answer the question, output reject, EOF."

    exe_correct = 0
    prog_correct = 0
    structure_errors = 0
    total = len(eval_dataset)
    
    total_tps = 0.0
    total_ttft = 0.0
    perf_count = 0

    for example in eval_dataset:
        try:
            # 1. Prompt Construction
            if set_name == "retrieved":
                retrieved_docs = retriever.retrieve_context(example, example["qa"]["question"], k=5)
                prompt = format_retrieved_prompt(example, retrieved_docs)
            elif set_name == "gold_chunks":
                prompt = format_gold_chunks_prompt(example)
            elif set_name == "gold":
                prompt = format_gold_prompt(example)
            else:
                # RGB testbeds have pre-computed context
                prompt = f"Context:\n{example['qa']['eval_context']}\n\nQuestion:\n{example['qa']['question']}\n"

            # 2. Generation & Timing
            res = client.generate(prompt, system_prompt=sys_prompt)
            raw_response = res["text"]
            cleaned_response = clean_llm_response(raw_response)
            pred_tokens = program_tokenization(cleaned_response)

            if res.get("tps", 0) > 0:
                total_tps += res["tps"]
                total_ttft += res["ttft"]
                perf_count += 1
        except Exception:
            pred_tokens = ["error", "EOF"]
            cleaned_response = "error"

        gold_program = example["qa"]["program"]
        gold_exe_ans = example["qa"]["exe_ans"]
        table = example["table"]
        gold_tokens = program_tokenization(gold_program)

        # 3. Structure error validation
        is_structure_error = False
        if len(pred_tokens) > 0 and pred_tokens[0] not in ["reject", "error"]:
            for t_idx, token in enumerate(pred_tokens[:-1]):
                if t_idx % 4 == 0:
                    if token.strip("(") not in ["add", "subtract", "multiply", "divide", "exp", "greater", 
                                               "table_max", "table_min", "table_sum", "table_average"]:
                        is_structure_error = True
                        break
                if (t_idx + 1) % 4 == 0:
                    if token != ")":
                        is_structure_error = True
                        break
        if is_structure_error or "error" in pred_tokens:
            structure_errors += 1

        # 4. Program accuracy
        if set_name != "negative_rejection":
            if equal_program(gold_tokens, pred_tokens):
                prog_correct += 1

        # 5. Execution accuracy
        if set_name == "negative_rejection":
            if len(pred_tokens) > 0 and pred_tokens[0] == "reject":
                exe_correct += 1
        else:
            try:
                invalid_flag, pred_exe_val = eval_program(pred_tokens, table)
                if invalid_flag == 0 and pred_exe_val == gold_exe_ans:
                    exe_correct += 1
            except Exception:
                pass

    return {
        "total": total,
        "exe_acc": exe_correct / total if total > 0 else 0.0,
        "prog_acc": prog_correct / total if total > 0 else 0.0,
        "struct_err": structure_errors / total if total > 0 else 0.0,
        "avg_tps": total_tps / perf_count if perf_count > 0 else 0.0,
        "avg_ttft": total_ttft / perf_count if perf_count > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Structured Benchmarking Harness")
    parser.add_argument("--limit", type=int, default=20, help="Limit for standard evaluations")
    parser.add_argument("--rgb-limit", type=int, default=50, help="Limit for RGB testbed evaluations")
    parser.add_argument("--aws-region", type=str, default="us-east-1", help="AWS Region for SageMaker")
    args = parser.parse_args()

    test_path = os.path.join(PROJECT_ROOT, "data", "test.json")

    # 1. Ensure clean dataset splits exist
    if not os.path.exists(test_path):
        print("[*] data/test.json not found. Running scripts/preprocess.py...")
        try:
            from scripts.preprocess import main as run_preprocess
            run_preprocess()
        except Exception as e:
            print(f"[Error] Failed to run preprocess.py: {e}")
            sys.exit(1)

    # 2. Check if rgb_eval folder and files exist, otherwise generate them
    rgb_dir = os.path.join(PROJECT_ROOT, "data", "rgb_eval")
    rgb_testbeds = {
        "noise_robustness": os.path.join(rgb_dir, "noise_robustness.json"),
        "negative_rejection": os.path.join(rgb_dir, "negative_rejection.json"),
        "information_integration": os.path.join(rgb_dir, "information_integration.json"),
        "counterfactual_robustness": os.path.join(rgb_dir, "counterfactual_robustness.json")
    }
    
    missing_testbeds = [name for name, path in rgb_testbeds.items() if not os.path.exists(path)]
    if missing_testbeds:
        print(f"[*] Some RGB testbed files ({', '.join(missing_testbeds)}) are missing. Generating them...")
        os.makedirs(rgb_dir, exist_ok=True)
        try:
            from evaluation.generate_rgb import (
                generate_noise_robustness,
                generate_negative_rejection,
                generate_information_integration,
                generate_counterfactual_robustness
            )
            with open(test_path, "r") as f:
                data = json.load(f)
            
            import random
            random.seed(42)
            
            if "noise_robustness" in missing_testbeds:
                with open(rgb_testbeds["noise_robustness"], "w") as f:
                    json.dump(generate_noise_robustness(data, limit=50, select_random=False), f, indent=4)
            if "negative_rejection" in missing_testbeds:
                with open(rgb_testbeds["negative_rejection"], "w") as f:
                    json.dump(generate_negative_rejection(data, limit=50, select_random=False), f, indent=4)
            if "information_integration" in missing_testbeds:
                with open(rgb_testbeds["information_integration"], "w") as f:
                    json.dump(generate_information_integration(data, limit=50, select_random=False), f, indent=4)
            if "counterfactual_robustness" in missing_testbeds:
                with open(rgb_testbeds["counterfactual_robustness"], "w") as f:
                    json.dump(generate_counterfactual_robustness(data, limit=50, select_random=False), f, indent=4)
            print("[*] RGB testbeds successfully generated!")
        except Exception as e:
            print(f"[Error] Failed to generate RGB testbeds: {e}")
            sys.exit(1)

    # Pre-flight endpoint connectivity checks
    ollama_running = is_port_open(11434)
    mlx_running = is_port_open(11435)

    # Check if a SageMaker endpoint is active (checks if AWS credentials allow listing/connecting)
    sagemaker_endpoint_active = False
    try:
        import boto3
        sm_client = boto3.client("sagemaker", region_name=args.aws_region)
        endpoints = sm_client.list_endpoints(NameContains="finqa-llama3-2-tuned-endpoint", StatusEquals="InService")
        if len(endpoints.get("Endpoints", [])) > 0:
            sagemaker_endpoint_active = True
    except Exception:
        pass

    print("================================================================================")
    print(" PRE-FLIGHT ENDPOINT STATUS CHECK")
    print("================================================================================")
    print(f"Ollama Server (Port 11434):  {'ACTIVE (Baseline model available)' if ollama_running else 'OFFLINE (Base evaluation skipped)'}")
    print(f"MLX-LM Server (Port 11435):  {'ACTIVE (Tuned model available)' if mlx_running else 'OFFLINE (Tuned evaluation skipped)'}")
    print(f"AWS SageMaker Endpoint:      {'ACTIVE (Cloud served model available)' if sagemaker_endpoint_active else 'OFFLINE (SageMaker evaluation skipped)'}")
    print("================================================================================\n")

    if not ollama_running and not mlx_running and not sagemaker_endpoint_active:
        print("[Critical Error] No serving endpoints (Ollama, MLX, or SageMaker) are reachable. Exiting.")
        sys.exit(1)

    models_to_test = []
    if ollama_running:
        models_to_test.append(("Baseline (finqa-llama3.2)", "finqa-llama3.2"))
    if mlx_running:
        models_to_test.append(("Fine-Tuned (finqa-llama3.2-tuned)", "finqa-llama3.2-tuned"))
    if sagemaker_endpoint_active:
        models_to_test.append(("SageMaker Fine-Tuned (sagemaker-endpoint)", "finqa-llama3.2-sagemaker"))

    results = {}
    standard_modes = ["gold_chunks", "retrieved"]
    
    for model_label, model_name in models_to_test:
        print(f"[*] Starting benchmark suite for model: {model_label}...")
        client = OllamaClient(model=model_name, aws_region=args.aws_region)
        model_results = {}

        # Standard splits
        for mode in standard_modes:
            print(f"  - Running standard mode: '{mode}' (limit={args.limit})...")
            mode_res = run_benchmark_set(test_path, args.limit, client, mode)
            model_results[mode] = mode_res

        # RGB Testbeds
        for testbed_name, path in rgb_testbeds.items():
            print(f"  - Running RGB testbed: '{testbed_name}' (limit={args.rgb_limit})...")
            testbed_res = run_benchmark_set(path, args.rgb_limit, client, testbed_name)
            model_results[testbed_name] = testbed_res

        results[model_name] = model_results

    # Print summary table to console
    print("\n" + "="*112)
    print("                                     FINQA COMPARATIVE BENCHMARK SUMMARY")
    print("="*112)
    print(f"{'Evaluation Set / Failure Mode':<30} | {'Model':<12} | {'Count':<5} | {'Prog Acc':<8} | {'Exe Acc':<8} | {'Struct Err':<10} | {'Avg TPS':<8} | {'Avg TTFT':<8}")
    print("-"*112)

    all_keys = ["gold_chunks", "retrieved", "noise_robustness", "negative_rejection", "information_integration", "counterfactual_robustness"]
    
    for key in all_keys:
        key_label = key.replace("_", " ").title()
        for model_name in results.keys():
            if key not in results[model_name]:
                continue
            metrics = results[model_name][key]
            
            prog_str = f"{metrics['prog_acc']*100:.1f}%"
            exe_str = f"{metrics['exe_acc']*100:.1f}%"
            
            if key == "negative_rejection":
                prog_str = "N/A"
                exe_str = f"{metrics['exe_acc']*100:.1f}% (Rej)"
                
            model_label = "Baseline"
            if "tuned" in model_name:
                model_label = "Fine-Tuned"
            elif "sagemaker" in model_name:
                model_label = "SageMaker"
                
            print(f"{key_label:<30} | {model_label:<12} | {metrics['total']:<5} | {prog_str:<8} | {exe_str:<8} | {metrics['struct_err']*100:.1f}% | {metrics['avg_tps']:<8.2f} | {metrics['avg_ttft']:<7.3f}s")
        print("-" * 112)
    print("="*112)

    # Save outputs to file under docs/
    docs_dir = os.path.join(PROJECT_ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # JSON Report
    report_json_path = os.path.join(docs_dir, "benchmark_results.json")
    json_data = {
        "timestamp": timestamp,
        "limits": {"standard": args.limit, "rgb": args.rgb_limit},
        "results": results
    }
    with open(report_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"\n[Info] Saved raw JSON report to: {report_json_path}")

    # Markdown Report
    report_md_path = os.path.join(docs_dir, "benchmark_results.md")
    with open(report_md_path, "w") as f:
        f.write(f"# FinQA Benchmark Evaluation Report\n\n")
        f.write(f"*Generated on:* `{timestamp}`\n\n")
        
        f.write(f"## Evaluation Configurations\n")
        f.write(f"* Standard Limits: `{args.limit}` examples\n")
        f.write(f"* RGB Testbed Limits: `{args.rgb_limit}` examples\n\n")
        
        f.write(f"## Benchmark Results Summary\n\n")
        f.write(f"| Evaluation Set | Model Type | Examples | Program Acc | Execution Acc | Structure Error | Avg TPS | Avg TTFT |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for key in all_keys:
            key_label = key.replace("_", " ").title()
            for model_name in results.keys():
                metrics = results[model_name][key]
                prog_str = f"{metrics['prog_acc']*100:.1f}%"
                exe_str = f"{metrics['exe_acc']*100:.1f}%"
                if key == "negative_rejection":
                    prog_str = "N/A"
                    exe_str = f"{metrics['exe_acc']*100:.1f}% (Refusal)"
                model_label = "Baseline"
                if "tuned" in model_name:
                    model_label = "Fine-Tuned"
                elif "sagemaker" in model_name:
                    model_label = "SageMaker"
                f.write(f"| {key_label} | {model_label} | {metrics['total']} | {prog_str} | {exe_str} | {metrics['struct_err']*100:.1f}% | {metrics['avg_tps']:.2f} | {metrics['avg_ttft']:.3f}s |\n")
        
    print(f"[Info] Saved formatted Markdown report to: {report_md_path}")


if __name__ == "__main__":
    main()
