"""
This script serves as a local data provider for Type 3/4 Java clones. 
It shuffles and streams pairs of source code files and their labels 
via a FastAPI endpoint, allowing the XGBoost.py script to evaluate 
model performance on custom local datasets.
"""

import os
import csv
import random
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER = "../data/type5_clones"  
SOURCE_FOLDER = os.path.join(DATA_FOLDER, "id2sourcecode")
TYPE5_CSV = os.path.join(DATA_FOLDER, "clone.csv")
NONCLONE_CSV = os.path.join(DATA_FOLDER, "nonclone.csv")

def read_java_file(file_id):
    """Helper to read a .java file by ID from the id2sourcecode subfolder"""
    filepath = os.path.join(SOURCE_FOLDER, f"{file_id}.java") 
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    return None

def generate_pairs(split: str):
    """
    Generator that yields JSON lines for the client.
    split: 'train' or 'val'
    """
    pairs = []
    
    # 1. Load Clone Metadata
    if os.path.exists(TYPE5_CSV):
        with open(TYPE5_CSV, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0], row[1], 1))

    # 2. Load Non-Clone Metadata
    if os.path.exists(NONCLONE_CSV):
        with open(NONCLONE_CSV, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0], row[1], 0))

    # 3. Shuffle & Split
    random.seed(42)
    random.shuffle(pairs)
    
    split_index = int(len(pairs) * 0.8)
    
    if split == 'train':
        target_pairs = pairs[:split_index]
    else:
        target_pairs = pairs[split_index:]

    # 4. Stream Data
    for p in target_pairs:
        id1, id2, label = p
        code1 = read_java_file(id1)
        code2 = read_java_file(id2)

        if code1 and code2:
            data = {
                "id1": id1,
                "id2": id2,
                "func1": code1,
                "func2": code2,
                "label": label
            }
            yield json.dumps(data) + "\n"

@app.get("/dataset")
async def get_dataset(split: str = "train"):
    """
    Endpoint: GET /dataset?split=train
    Returns: A stream of JSON objects
    """
    return StreamingResponse(generate_pairs(split), media_type="application/x-json-stream")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Data Service in {DATA_FOLDER}...")
    uvicorn.run(app, host="127.0.0.1", port=8000)