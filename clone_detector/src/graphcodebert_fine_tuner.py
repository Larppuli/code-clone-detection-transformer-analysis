"""
This script optimizes the GraphCodeBERT encoder for clone detection by 
incorporating both source code and Data Flow Graphs (DFG). It supports 
multi-language tuning (Java/Python) and different pooling strategies 
([CLS] vs. Mean) to produce embeddings for the XGBoost.py classifier.
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
MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LEN = 512
BATCH_SIZE = 4       
ACCUMULATION_STEPS = 2    
EPOCHS = 4             
LR = 2e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
TUNING_SIZE = 4000

# Path to local graph_parser
REPO_PATH = os.path.abspath("./CodeBERT/GraphCodeBERT/codesearch")
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Ask user for language and pooling choices
while True:
    while True:
        lang_choice = input("Select fine-tuning for Java (0) or for Python (1): ").strip()
        if lang_choice in {"0", "1"}:
            break
        print("Invalid input. Please enter 0 (Java) or 1 (Python).")
    pooling_choice = input("Select [CLS] pooling (0) or Mean pooling (1): ").strip()
    if pooling_choice in {"0", "1"}:
        fine_tuner_suffix = "cls" if pooling_choice == "0"  else "mean"
        break
    print("Invalid input. Please enter 0 (Java) or 1 (Python).")

# Initialize all variables based on the choice
if lang_choice == "1":
    # Python Configuration
    DATASET_NAME = "PoolC/1-fold-clone-detection-600k-5fold"
    language = "python"
    label_key = "similar"
    code_var_names = ["code1", "code2"]
    SAVE_DIR = f"../fine_tuned_models/graphcodebert-clone-python-{fine_tuner_suffix}"
    val_key = "train"
    print("Preparing Python fine-tuning...")
else:
    # Java Configuration
    DATASET_NAME = "google/code_x_glue_cc_clone_detection_big_clone_bench"
    language = "java"
    label_key = "label"
    code_var_names = ["func1", "func2"]
    SAVE_DIR = f"../fine_tuned_models/graphcodebert-clone-java-{fine_tuner_suffix}"
    val_key = "validation"
    print("Preparing Java fine-tuning...")

# --- MODEL DEFINITION ---
class GraphCodeBERTCloneModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(2305, 2)

    def mean_pooling(self, model_output, attention_mask):
        """Calculates the average of token embeddings, ignoring padding."""
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, ids1, mask1, ids2, mask2, labels=None):
        if pooling_choice == "0":
            # Extract the [CLS] token embedding from the last hidden state for both code fragments
            u = self.encoder(ids1, attention_mask=mask1)[0][:, 0, :]
            v = self.encoder(ids2, attention_mask=mask2)[0][:, 0, :]
        else:
            # Pass code snippets through the GraphCodeBERT encoder to get full token-level hidden states
            out1 = self.encoder(ids1, attention_mask=mask1)
            out2 = self.encoder(ids2, attention_mask=mask2)
            
            # Calculate the average of all token embeddings in the sequence, excluding padding tokens
            u = self.mean_pooling(out1, mask1) 
            v = self.mean_pooling(out2, mask2)
        
        dot_product = torch.sum(u * v, dim=-1, keepdim=True) 
        abs_diff = torch.abs(u - v)           
        
        fused_vector = torch.cat([u, v, abs_diff, dot_product], dim=-1)
        
        binary_logits = self.classifier(fused_vector)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(binary_logits, labels)
            
        return {"loss": loss, "logits": binary_logits}

# --- DATASET DEFINITION ---
class CloneDataset(Dataset):
    def __init__(self, data, tokenizer, language, limit=None):
        self.tokenizer = tokenizer
        self.language = language
        self.raw_data = data
        
        # Identify non-truncated indices up to the requested limit
        self.valid_indices = self._get_clean_indices(limit)

    def _get_clean_indices(self, limit):
        clean_ids = []
        print(f"Filtering dataset for non-truncated pairs (Target: {limit if limit else 'All'})...")
        
        for i in tqdm(range(len(self.raw_data))):
            if is_not_truncated(self.raw_data[i], self.tokenizer):
                clean_ids.append(i)
            
            if limit and len(clean_ids) >= limit:
                break
        return clean_ids

    def extract_dataflow(self, code):
        from tree_sitter import Parser, Language
        from graph_parser import DFG_python, DFG_java, remove_comments_and_docstrings, tree_to_token_index, index_to_code_token
        
        local_parser = Parser()
        so_path = './CodeBERT/GraphCodeBERT/codesearch/graph_parser/my-languages.so'
        local_parser.set_language(Language(so_path, self.language))
        
        dfg_fn = DFG_python if self.language == 'python' else DFG_java
        
        try:
            code = remove_comments_and_docstrings(code, self.language)
            tree = local_parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            dfg, _ = dfg_fn(root_node, index_to_code_token, code)
            return code_tokens, dfg
        except Exception:
            return [], []

    def process_code(self, code):
        code_tokens, dfg = self.extract_dataflow(code)
        code_tokens = code_tokens[:440] 
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        dfg_tokens = [x[0] for x in dfg[:64]]
        all_tokens = tokens + dfg_tokens
        ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        padding_len = MAX_LEN - len(ids)
        ids += [self.tokenizer.pad_token_id] * padding_len
        mask = [1] * len(all_tokens) + [0] * padding_len
        return torch.tensor(ids), torch.tensor(mask)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.raw_data[actual_idx]
        
        ids1, mask1 = self.process_code(item[code_var_names[0]])
        ids2, mask2 = self.process_code(item[code_var_names[1]])
        return {
            "ids1": ids1, "mask1": mask1,
            "ids2": ids2, "mask2": mask2,
            "label": torch.tensor(int(item[label_key]), dtype=torch.long)
        }

    def __len__(self):
        return len(self.valid_indices)
    
def is_not_truncated(item, tokenizer, max_len=512, dfg_limit=64):
    """
    Checks if both code fragments in a pair fit within the MAX_LEN limit.
    GraphCodeBERT uses: 1 [CLS] + Code Tokens + 1 [SEP] + DFG tokens.
    """
    # Reserved space for special tokens and DFG
    reserved = 2 + dfg_limit 
    max_code_tokens = max_len - reserved

    for var_name in code_var_names:
        code = item[var_name]
        # Fast tokenization to count length without full processing
        tokens = tokenizer.tokenize(code)
        if len(tokens) > max_code_tokens:
            return False
    return True

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset(DATASET_NAME)

    train_set = CloneDataset(dataset['train'], tokenizer, language, limit=TUNING_SIZE)
    
    last_train_idx = train_set.valid_indices[-1] + 1
    
    if val_key == "train":
        # Python Case: Slice the training data starting after the last training sample
        val_raw = dataset['train'].select(range(last_train_idx, len(dataset['train'])))
        print(f"Creating Python validation split from dataset['train'] starting at index {last_train_idx}")
    else:
        # Java Case: Use the provided validation split
        val_raw = dataset[val_key]
        print("Using provided Java validation split")
    
    val_set = CloneDataset(val_raw, tokenizer, language, limit=int(TUNING_SIZE * 0.1))
    
    # --- Rest of the function remains the same ---
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        pin_memory=False
    )
    
    model = GraphCodeBERTCloneModel(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)

    total_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(loop):
            out = model(
                batch["ids1"].to(DEVICE), batch["mask1"].to(DEVICE),
                batch["ids2"].to(DEVICE), batch["mask2"].to(DEVICE),
                batch["label"].to(DEVICE)
            )
            loss = out["loss"] / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            loop.set_postfix(loss=out["loss"].item())

        val_loss = validate(model, val_loader)
        print(f"\nEpoch {epoch+1} done. Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.encoder.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

    print(f"Complete in {str(timedelta(seconds=int(time.time() - total_start)))}")

if __name__ == "__main__":
    train()