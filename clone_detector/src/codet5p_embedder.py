"""
This script converts source code snippets into embeddings using CodeT5+.
It supports local FastAPI streaming and Hugging Face datasets (Java/Python) and 
saves the resulting vector pairs to .jsonl files for training XGBoost.py classifier.
"""

import os
import sys
import json
import torch
import requests
import subprocess
import time
import socket
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------------------------
# SETUP PATHS & CONFIG
# -----------------------------------------------------------------------------

DATASET_SIZE = 50000
MAX_LEN = 512

# Select dataset
while True:
    choice = input("Select source: Java (0), Python (1), or Local FastAPI (2): ").strip()
    if choice in {"0", "1", "2"}:
        break
    print("Invalid input. Please enter 0, 1, or 2.")

# Dataset configuration
if choice == "0":
    DATASET_NAME = "google/code_x_glue_cc_clone_detection_big_clone_bench"
    label_name, code_var_names, test_split, mode, language = "label", ["func1", "func2"], "test", "hf", "java"
elif choice == "1":
    DATASET_NAME = "PoolC/1-fold-clone-detection-600k-5fold"
    label_name, code_var_names, test_split, mode, language = "similar", ["code1", "code2"], "val", "hf", "python"
else:
    API_URL = "http://127.0.0.1:8000/dataset"
    label_name, code_var_names, test_split, mode, language = "label", ["func1", "func2"], "val", "api", "java"


# Import comment removal tool
REPO_PATH = os.path.abspath("./CodeBERT/GraphCodeBERT/codesearch")
sys.path.insert(0, REPO_PATH)
try:
    from graph_parser import remove_comments_and_docstrings
    HAS_PARSER = True
    print("Comment removal tool imported successfully.")
except ImportError:
    HAS_PARSER = False

# -----------------------------------------------------------------------------
# Service Manager for FastApi
# -----------------------------------------------------------------------------
def is_port_open(port, host='127.0.0.1'):
    """Check if the FastAPI service is listening on the expected port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_api_service():
    """Command data_service.py to run in a background process."""
    if is_port_open(8000):
        return None

    print("Launching local Data Service...")
    process = subprocess.Popen(
        [sys.executable, "data_service.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )

    timeout = 20
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open(8000):
            return process
        time.sleep(1)
    
    process.terminate()
    raise ConnectionError("Timeout: Data service failed to start within 20 seconds.")

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

BASE_MODEL_NAME = "Salesforce/codet5p-110m-embedding"

# Ask for base or fine-tuned model
model_choice = input("Select base model (0) or fine-tuned model (1): ").strip()
if model_choice == "0":
    model_path = BASE_MODEL_NAME
    suffix = "base"
else:
    model_path = f"../fine_tuned_models/codet5p-clone-{language}"
    suffix = "tuned"

print(f"Loading weights from: {model_path}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# Load the model weights (either base or your fine-tuned ones)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model.to(DEVICE)
model.eval()

# Output file locations
prefix = language if choice in {"0", "1"} else "fastapi"
TRAIN_SAVE_LOC = f"../data/embeddings/CodeT5P/{prefix}_codet5p_embeddings_{suffix}.jsonl"
VAL_SAVE_LOC = f"../data/embeddings/CodeT5P/{prefix}_codet5p_embeddings_{suffix}_val.jsonl"
os.makedirs(os.path.dirname(TRAIN_SAVE_LOC), exist_ok=True)

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------

class RemoteDatasetStream:
    """Makes local FastAPI streams act like Hugging Face datasets."""
    def __init__(self, url):
        self.url = url
    
    def __iter__(self):
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line.decode('utf-8'))
    
    def shuffle(self, **kwargs):
        return self

def get_embedding(code_snippet):
    """Generates embedding for a single code snippet with shape-aware pooling"""
    
    # 1. Pre-process
    if HAS_PARSER:
        try:
            code_snippet = remove_comments_and_docstrings(code_snippet, language)
        except Exception:
            pass
            
    # 2. Tokenize
    inputs = tokenizer(
        code_snippet, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LEN
    ).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        hidden = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # 4. Shape-Aware Pooling
        if len(hidden.shape) == 3:
            # Manual Mean Pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze(0)
        else:
            embedding = hidden.squeeze(0)
            
    return embedding.cpu().tolist()

def would_truncate(code_str):
    """Check if code exceeds the model's 512 token limit"""
    tokens = tokenizer(code_str, truncation=False)["input_ids"]
    return len(tokens) > MAX_LEN

def process_and_save(dataset, output_file, target_total, start_index=0):
    """Generates 50/50 balanced embeddings and saves to JSONL"""
    target_per_label = target_total // 2
    count_0, count_1 = 0, 0
    
    with open(output_file, 'w') as f:
        pbar = tqdm(total=target_total, desc=f"Saving to {os.path.basename(output_file)}")
        
        for i, example in enumerate(dataset):
            if count_0 >= target_per_label and count_1 >= target_per_label:
                break
            if i < start_index:
                continue

            label = int(example[label_name])
            
            # Label balancing
            if (label == 0 and count_0 >= target_per_label) or (label == 1 and count_1 >= target_per_label):
                continue

            # Truncation check
            if would_truncate(example[code_var_names[0]]) or would_truncate(example[code_var_names[1]]):
                continue 

            try:
                emb1 = get_embedding(example[code_var_names[0]])
                emb2 = get_embedding(example[code_var_names[1]])
                
                record = {
                    "label": label,
                    "embedding1": emb1,
                    "embedding2": emb2
                }
                f.write(json.dumps(record) + "\n")
                
                if label == 0: count_0 += 1
                else: count_1 += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing item {i}: {e}")
                continue

        pbar.close()
    return i

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    service_proc = None
    if mode == "api":
        try:
            service_proc = start_api_service()
            print(f"Connecting to Local Data Service: {API_URL}")
            ds_val = RemoteDatasetStream(f"{API_URL}?split=val")
            ds = None
        except Exception as e:
            print(f"Failed to initialize local service: {e}")
            sys.exit(1)
    else:
        # Standard Hugging Face Loading
        print(f"Loading Hugging Face dataset: {DATASET_NAME}")
        ds = load_dataset(DATASET_NAME, split='train', streaming=True).shuffle(buffer_size=10000, seed=42)
        ds_val = load_dataset(DATASET_NAME, split=test_split, streaming=True).shuffle(buffer_size=10000, seed=42)

    try:
        # Step 2: Generate Training Set
        if mode != "api":
            print("\n--- Starting Training Set Generation ---")
            process_and_save(
                dataset=ds, 
                output_file=TRAIN_SAVE_LOC, 
                target_total=DATASET_SIZE, 
                start_index=0
            )
        else:
            print("\nSkipping Training Set Generation (FastAPI mode: Validation Only)")

        # Step 3: Generate Validation Set
        VAL_SIZE = int(DATASET_SIZE * 0.2)
        
        print(f"\n--- Starting Validation Set Generation ({VAL_SIZE} items) ---")
        process_and_save(
            dataset=ds_val, 
            output_file=VAL_SAVE_LOC, 
            target_total=VAL_SIZE, 
            start_index=0
        )

    finally:
        # Step 4: Graceful Shutdown
        if service_proc:
            print("\nTerminating background Data Service...")
            service_proc.terminate()
            try:
                service_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service_proc.kill()