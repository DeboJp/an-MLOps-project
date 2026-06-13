import os
import json

def clean_dataset(input_path, output_path):
    """
    Loads raw FinQA dataset, discards the heavy BM25/TF-IDF retrieval metadata,
    and keeps a clean nested 'qa' structure that is 100% compatible with evaluator.py.
    """
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Processing: {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for example in data:
        qa = example.get("qa", {})
        
        # Keep only the core fields in qa to remain compatible with evaluation metrics
        clean_qa = {
            "question": qa.get("question", ""),
            "program": qa.get("program", None),
            "exe_ans": qa.get("exe_ans", None),
            "gold_inds": qa.get("gold_inds", {})
        }
        
        # Build clean example dictionary
        clean_item = {
            "id": example.get("id"),
            "pre_text": example.get("pre_text", []),
            "post_text": example.get("post_text", []),
            "table": example.get("table", []),
            "qa": clean_qa
        }
        cleaned_data.append(clean_item)

    # Save cleaned JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)

    # Calculate size differences
    orig_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    clean_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to: {output_path}")
    print(f"Size: {orig_size_mb:.2f} MB -> {clean_size_mb:.2f} MB (Reduced by {(1 - clean_size_mb/orig_size_mb)*100:.1f}%)\n")

def main():
    # Define directories relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "FinQA-main", "dataset")
    clean_dir = os.path.join(project_root, "data")
    
    # Create output directory
    os.makedirs(clean_dir, exist_ok=True)
    
    # Map files
    files = {
        "train.json": "train.json",
        "dev.json": "val.json",
        "test.json": "test.json",
        "private_test.json": "private_test.json"
    }
    
    for filename, output_name in files.items():
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(clean_dir, output_name)
        clean_dataset(input_path, output_path)

if __name__ == "__main__":
    main()
