"""
This script converts source code snippets into embeddings using Llama 3.2 1B.
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
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# SETUP & GATED ACCESS
# -----------------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_SIZE = 50000
MAX_LEN = 2048 

# 1. Select Source
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
    print("Warning: graph_parser not found. Comments will not be removed.")

# -----------------------------------------------------------------------------
# SERVICE MANAGER & UTILITIES
# -----------------------------------------------------------------------------
def is_port_open(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_api_service():
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
    raise ConnectionError("Timeout: Data service failed to start.")

class RemoteDatasetStream:
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

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"

print(f"Loading base Llama weights from: {BASE_MODEL_NAME} on {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, 
    token=HF_TOKEN,
    dtype=torch.float16 if DEVICE != torch.device("cpu") else torch.float32,
    trust_remote_code=True
)
model.to(DEVICE)
model.eval()

# Output file locations
prefix = language if mode == "hf" else "fastapi"
TRAIN_SAVE_LOC = f"../data/embeddings/Llama/{prefix}_llama_embeddings_base.jsonl"
VAL_SAVE_LOC = f"../data/embeddings/Llama/{prefix}_llama_embeddings_base_val.jsonl"
os.makedirs(os.path.dirname(TRAIN_SAVE_LOC), exist_ok=True)

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def get_embedding(code_snippet):
    if HAS_PARSER:
        try:
            code_snippet = remove_comments_and_docstrings(code_snippet, language)
        except Exception: pass
            
    inputs = tokenizer(code_snippet, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] 
        
        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        embedding = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        
    return embedding.squeeze(0).float().cpu().tolist()

def would_truncate(code_str):
    tokens = tokenizer(code_str, truncation=False)["input_ids"]
    return len(tokens) > MAX_LEN

def process_and_save(dataset, output_file, target_total, start_index=0):
    target_per_label = target_total // 2
    count_0, count_1 = 0, 0
    
    with open(output_file, 'w') as f:
        pbar = tqdm(total=target_total, desc=f"Saving to {os.path.basename(output_file)}")
        try:
            for i, example in enumerate(dataset):
                if i < start_index:
                    continue

                if count_0 >= target_per_label and count_1 >= target_per_label: 
                    break
                
                label = int(example[label_name])
                if (label == 0 and count_0 >= target_per_label) or (label == 1 and count_1 >= target_per_label):
                    continue

                if would_truncate(example[code_var_names[0]]) or would_truncate(example[code_var_names[1]]):
                    continue 

                try:
                    emb1, emb2 = get_embedding(example[code_var_names[0]]), get_embedding(example[code_var_names[1]])
                    f.write(json.dumps({"label": label, "embedding1": emb1, "embedding2": emb2}) + "\n")
                    if label == 0: count_0 += 1
                    else: count_1 += 1
                    pbar.update(1)
                except Exception:
                    continue
            
            return i 
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\nStream interrupted or ended: {e}")
            return i
        finally:
            pbar.close()

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
            print(f"Failed to initialize local service: {e}"); sys.exit(1)
    else:
        ds = load_dataset(DATASET_NAME, split='train', streaming=True).shuffle(buffer_size=10000, seed=42)
        ds_val = load_dataset(DATASET_NAME, split=test_split, streaming=True).shuffle(buffer_size=10000, seed=42)

    try:
        # Step 1: Training Set Generation
        last_index = 0
        if mode != "api":
            print("\n--- Starting Training Set Generation ---")
            last_index = process_and_save(ds, TRAIN_SAVE_LOC, DATASET_SIZE, 0)
        else:
            print("\n⏭ Skipping Training Set Generation (FastAPI mode selected)")

        # Step 2: Validation Set Generation
        start_idx = last_index + 1 if mode != "api" else 0
        VAL_SIZE = int(DATASET_SIZE * 0.2)
        
        print(f"\n--- Starting Validation Set Generation ({VAL_SIZE} items) ---")
        process_and_save(ds_val, VAL_SAVE_LOC, VAL_SIZE, start_idx)
        
    finally:
        if service_proc:
            print("\nTerminating background Data Service...")
            service_proc.terminate()
            try: service_proc.wait(timeout=5)
            except subprocess.TimeoutExpired: service_proc.kill()