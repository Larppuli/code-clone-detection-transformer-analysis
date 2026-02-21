"""
Fine-tunes the CodeT5+ encoder using CosineEmbeddingLoss to improve 
semantic similarity representation for Java and Python code clones. 
The resulting model is used to generate embeddings for the XGBoost.py classifier.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import time
from datetime import timedelta

# --- GLOBALS ---
MODEL_NAME = "Salesforce/codet5p-110m-embedding"
MAX_LEN = 512
BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
TUNING_SIZE = 4000

REPO_PATH = os.path.abspath("./CodeBERT/GraphCodeBERT/codesearch")

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Ask user for language choices
while True:
    lang_choice = input("Select fine-tuning for Java (0) or for Python (1): ").strip()
    if lang_choice in {"0", "1"}:
        break
    print("Invalid input. Please enter 0 (Java) or 1 (Python).")

# Initialize variables based on choice
if lang_choice == "1":
    # Python Configuration
    DATASET_NAME = "PoolC/1-fold-clone-detection-600k-5fold"
    language = "python"
    label_key = "similar"
    code_var_names = ["code1", "code2"]
    SAVE_DIR = "../fine_tuned_models/codet5p-clone-python"
    val_key = "train"
    print("Preparing Python fine-tuning...")
else:
    # Java Configuration
    DATASET_NAME = "google/code_x_glue_cc_clone_detection_big_clone_bench"
    language = "java"
    label_key = "label"
    code_var_names = ["func1", "func2"]
    SAVE_DIR = "../fine_tuned_models/codet5p-clone-java"
    val_key = "validation"
    print("Preparing Java fine-tuning...")

# --- MODEL DEFINITION ---
class CodeT5CloneModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Ue CosineEmbeddingLoss for Siamese training
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, ids1, mask1, ids2, mask2, labels=None):
            # 1. Encode first code snippet
            hidden1 = self.encoder(input_ids=ids1, attention_mask=mask1)
            
            if len(hidden1.shape) == 3:
                mask1_expanded = mask1.unsqueeze(-1).float()
                sum_embeddings1 = torch.sum(hidden1 * mask1_expanded, dim=1)
                sum_mask1 = torch.clamp(mask1_expanded.sum(dim=1), min=1e-9)
                e1 = sum_embeddings1 / sum_mask1
            else:
                e1 = hidden1

            # 2. Encode second code snippet
            hidden2 = self.encoder(input_ids=ids2, attention_mask=mask2)
            
            if len(hidden2.shape) == 3:
                mask2_expanded = mask2.unsqueeze(-1).float()
                sum_embeddings2 = torch.sum(hidden2 * mask2_expanded, dim=1)
                sum_mask2 = torch.clamp(mask2_expanded.sum(dim=1), min=1e-9)
                e2 = sum_embeddings2 / sum_mask2
            else:
                e2 = hidden2
            
            loss = None
            if labels is not None:
                target_labels = labels.clone()
                target_labels[target_labels == 0] = -1
                loss = self.loss_fn(e1, e2, target_labels)
                
            return {"loss": loss, "embedding_1": e1, "embedding_2": e2}

# --- DATASET DEFINITION ---
class CloneDataset(Dataset):
    def __init__(self, data, tokenizer, limit=None):
        self.tokenizer = tokenizer
        self.raw_data = data
        self.valid_indices = self._get_clean_indices(limit)

    def _get_clean_indices(self, limit):
        clean_ids = []
        print(f"Filtering dataset for non-truncated pairs (Target: {limit if limit else 'All'})...")
        
        for i in tqdm(range(len(self.raw_data))):
            if self._is_not_truncated(self.raw_data[i]):
                clean_ids.append(i)
            if limit and len(clean_ids) >= limit:
                break
        return clean_ids

    # Function to check if a code fragment would be truncated
    def _is_not_truncated(self, item):
        for var in code_var_names:
            if len(self.tokenizer(item[var], truncation=False)["input_ids"]) > MAX_LEN:
                return False
        return True

    def process_code(self, code):
            try:
                from graph_parser import remove_comments_and_docstrings
                code = remove_comments_and_docstrings(code, self.language)
            except ImportError:
                print("Warning: graph_parser not found, using raw code.")
            except Exception:
                pass

            inputs = self.tokenizer(
                code, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LEN
            )
            return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.raw_data[actual_idx]
        
        # Pass the code to process_code
        ids1, mask1 = self.process_code(item[code_var_names[0]])
        ids2, mask2 = self.process_code(item[code_var_names[1]])
        
        return {
            "ids1": ids1, "mask1": mask1,
            "ids2": ids2, "mask2": mask2,
            "label": torch.tensor(int(item[label_key]), dtype=torch.float)
        }

    def __len__(self):
        return len(self.valid_indices)

# --- TRAINING LOGIC ---
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                batch["ids1"].to(DEVICE), batch["mask1"].to(DEVICE),
                batch["ids2"].to(DEVICE), batch["mask2"].to(DEVICE),
                batch["label"].to(DEVICE)
            )
            total_loss += out["loss"].item()
    return total_loss / len(val_loader)

def train():
    # Load CodeT5+ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    dataset = load_dataset(DATASET_NAME)

    # Train Set
    train_set = CloneDataset(dataset['train'], tokenizer, limit=TUNING_SIZE)
    
    # Validation Splitting Logic
    last_train_idx = train_set.valid_indices[-1] + 1
    if val_key == "train":
        val_raw = dataset['train'].select(range(last_train_idx, len(dataset['train'])))
        print(f"Creating Python validation split from dataset['train'] starting at index {last_train_idx}")
    else:
        val_raw = dataset[val_key]
        print("Using provided Java validation split")
    
    val_set = CloneDataset(val_raw, tokenizer, limit=int(TUNING_SIZE * 0.1))
    
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = CodeT5CloneModel(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Starting training on {DEVICE}...")
    total_start = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(loop):
            if i % ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()

            out = model(
                batch["ids1"].to(DEVICE), batch["mask1"].to(DEVICE),
                batch["ids2"].to(DEVICE), batch["mask2"].to(DEVICE),
                batch["label"].to(DEVICE)
            )
            
            loss = out["loss"] / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
            
            loop.set_postfix(loss=out["loss"].item())

        val_loss = validate(model, val_loader)
        print(f"\nEpoch {epoch+1} done. Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.encoder.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"Saved new best model to {SAVE_DIR}")

    print(f"Complete in {str(timedelta(seconds=int(time.time() - total_start)))}")

if __name__ == "__main__":
    train()