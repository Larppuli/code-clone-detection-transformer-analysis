"""
Measures tokenizer's efficiency and "whitespace tax" across LLaMA, CodeT5+, and GraphCodeBERT 
tokenizers using BigCloneBench and PoolC-600k datasets.
"""

import os
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from prettytable import PrettyTable

# -----------------------------
# Configuration
# -----------------------------
HF_TOKEN = os.environ.get("hf_token")

MODELS = {
    "LLaMA 3.2 1B": "meta-llama/Llama-3.2-1B",
    "CodeT5+ (220M)": "Salesforce/codet5p-220m",
    "GraphCodeBERT": "microsoft/graphcodebert-base",
}

DATASETS = {
    "BigCloneBench": ("google/code_x_glue_cc_clone_detection_big_clone_bench", "func1", "func2"),
    "PoolC-600k": ("PoolC/1-fold-clone-detection-600k-5fold", "code1", "code2")
}

OUTPUT_FILE = "token_count.txt"

# -----------------------------
# Utility functions
# -----------------------------
def remove_whitespace(code_str):
    return re.sub(r'\s+', '', code_str) if isinstance(code_str, str) else ""

def get_token_counts(snippets, tokenizers, clean_whitespace=False):
    """Tokenizes snippets and returns counts per model."""
    results = {model_name: [] for model_name in tokenizers.keys()}
    for code in snippets:
        text = remove_whitespace(code) if clean_whitespace else code
        for model_name, tokenizer in tokenizers.items():
            try:
                tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                results[model_name].append(len(tokens))
            except Exception:
                results[model_name].append(0)
    return results

def summarize_counts(counts_dict):
    """Computes average token counts only."""
    summary = {}
    for model_name, counts in counts_dict.items():
        avg = sum(counts) / len(counts) if counts else 0
        summary[model_name] = {"Avg": avg}
    return summary

def print_compact_opposite_table(f, summary_clean, summary_raw, title):
    """PrettyTable showing whitespace-removed first, raw + % change second."""
    table = PrettyTable(["Model", "Whitespace Removed Avg", "Raw Avg", "% Change"])
    for model, clean_data in summary_clean.items():
        raw_data = summary_raw[model]
        avg_change = ((clean_data['Avg'] - raw_data['Avg']) / raw_data['Avg'] * 100) if raw_data['Avg'] else 0
        table.add_row([
            model,
            f"{clean_data['Avg']:.1f}",
            f"{raw_data['Avg']:.1f}",
            f"{avg_change:.2f}%"
        ])
    f.write(f"\n{title}\n")
    f.write(str(table) + "\n")

# -----------------------------
# Main
# -----------------------------
def main():
    # Load tokenizers
    tokenizers = {}
    for name, path in MODELS.items():
        print(f"Loading tokenizer for {name}...")
        tokenizers[name] = AutoTokenizer.from_pretrained(path, token=HF_TOKEN, trust_remote_code=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ds_name, (ds_path, col1, col2) in DATASETS.items():
            print(f"Processing dataset {ds_name}...")
            f.write(f"\n{'='*20} Dataset: {ds_name} {'='*20}\n")

            # Fetch snippets
            dataset = load_dataset(ds_path, split="train", streaming=True)
            iterator = iter(dataset)
            snippets = []
            while len(snippets) < 1000:
                try:
                    row = next(iterator)
                    if row[col1]: snippets.append(row[col1])
                    if len(snippets) < 1000 and row[col2]: snippets.append(row[col2])
                except StopIteration:
                    break
            snippets = snippets[:1000]

            # Token counts
            counts_raw = get_token_counts(snippets, tokenizers, clean_whitespace=False)
            counts_clean = get_token_counts(snippets, tokenizers, clean_whitespace=True)

            summary_raw = summarize_counts(counts_raw)
            summary_clean = summarize_counts(counts_clean)

            # Print compact "opposite" table
            print_compact_opposite_table(f, summary_clean, summary_raw, f"{ds_name} - Token Counts (Whitespace Removed First)")

    print(f"\n✅ Compact opposite tables saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
