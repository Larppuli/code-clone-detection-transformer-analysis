"""
Trains XGBoost models using pre-computed embeddings from GraphCodeBERT, CodeT5+, or Llama.
It evaluates performance on Java or Python code clones, producing ROC curves, 
Confusion Matrices, and detailed metric summaries.
"""

import os
import json
import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, 
                             confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef)
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from tabulate import tabulate 

# -----------------------------------------------------------------------------
# USER INPUTS & CONFIGURATION
# -----------------------------------------------------------------------------
while True:
    try:
        num_models = int(input("How many models do you want to compare? ").strip())
        if num_models > 0: break
    except ValueError:
        pass
    print("Please enter a valid positive integer.")

def get_model_config(index):
    print(f"\n--- Configure Model {index+1} ---")
    while True:
        m_choice = input(f"Select model: GraphCodeBERT (0), CodeT5+ (1) or Llama (2): ").strip()
        if m_choice in {"0", "1", "2"}: break
        print("Invalid input.")

    v_choice = "base"
    if m_choice != "2":
        while True: 
            v_choice = input(f"Select variant: Base (b) or Tuned (t): ").strip().lower()
            if v_choice in {"b", "t"}:
                v_choice = "tuned" if v_choice == 't' else "base"
                break
            else:
                print("Invalid input.")

    p_type = ""
    if m_choice == "0" and v_choice == "tuned":
        p_choice = input(f"Select pooling: CLS (0) or Mean (1): ").strip()
        p_type = "cls" if p_choice == "0" else "mean"

    return {"model_id": m_choice, "variant": v_choice, "pool": p_type}

configs = [get_model_config(i) for i in range(num_models)]

while True:
    dataset_choice = input("Select source: Java (0), Python (1), or Local FastAPI (2): ").strip()
    if dataset_choice in {"0", "1", "2"}:
        break
    print("Invalid input. Please enter 0, 1, or 2.")

model_map = {"0": "GraphCodeBERT", "1": "CodeT5+", "2": "Llama 3.2 1B"}
lang_map = {"0": "java", "1": "python", "2": "fastapi"}
selected_lang = lang_map[dataset_choice]

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def get_path_and_name(cfg, lang):
    m_name = model_map[cfg['model_id']]
    var = cfg['variant']
    pool = cfg['pool']
    
    if cfg['model_id'] == "0":
        suffix = f"_{pool}" if var == 'tuned' else ''
        path = f"../data/embeddings/GraphCodeBERT/{lang}_graphcodebert_embeddings_{var}{suffix}.jsonl"
    elif cfg['model_id'] == "1":
        path = f"../data/embeddings/CodeT5P/{lang}_codet5p_embeddings_{var}.jsonl"
    else:
        path = f"../data/embeddings/Llama/{lang}_llama_embeddings_base.jsonl"
    
    display_name = f"{m_name} {var}"
    if cfg['model_id'] == "0" and pool:
        display_name += f" {pool}"
    
    return path, display_name

def load_and_prepare(path, limit=None):
    if not os.path.exists(path): return None, None
    e1, e2, l = [], [], []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            e1.append(d['embedding1'])
            e2.append(d['embedding2'])
            l.append(d['label'])
            
    e1, e2 = np.array(e1, dtype=np.float32), np.array(e2, dtype=np.float32)
    X = np.hstack([e1, e2, np.abs(e1 - e2), np.sum(e1 * e2, axis=-1, keepdims=True)])
    return X, np.array(l)

# -----------------------------------------------------------------------------
# PROCESSING LOOP
# -----------------------------------------------------------------------------
output_dir = f"../data/comparison/"
os.makedirs(output_dir, exist_ok=True)

metrics_to_track = {}
roc_data = []
cm_data_list = [] 

for cfg in configs:
    # --- CROSS-DOMAIN LOGIC ---
    if selected_lang == "fastapi":
        # Train on Java
        path, base_name = get_path_and_name(cfg, "java")
        
        # Test on FastAPI Validation Set
        raw_val_path, _ = get_path_and_name(cfg, "fastapi")
        val_path = raw_val_path.replace(".jsonl", "_val.jsonl")
        
        name = base_name
    else:
        # Standard Within-Domain Logic
        path, name = get_path_and_name(cfg, selected_lang)
        val_path = path.replace(".jsonl", "_val.jsonl")

    X_train, y_train = load_and_prepare(path)
    X_val, y_val = load_and_prepare(val_path)

    if X_train is None:
        print(f"Skipping {name}: Path {path} not found.")
        continue
    if X_val is None:
        print(f"Skipping {name}: Test path {val_path} not found.")
        continue

    start_train = time.time()

    # Training Logic
    if cfg['model_id'] == "2":
        print(f"\n--- Training {name} (Fixed Params) ---")
        clf = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=-1
        )
        clf.fit(X_train, y_train)
    else:
        print(f"\n--- Training {name} (Randomized Search) ---")
        search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'),
            param_distributions={'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 6, 8]},
            n_iter=5, cv=2, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        clf = search.best_estimator_
    
    train_time = time.time() - start_train
    
    # Inference
    start_inf = time.time()
    y_prob = clf.predict_proba(X_val)[:, 1]
    inf_time = (time.time() - start_inf) / len(y_val) 
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics and Plots
    cm = confusion_matrix(y_val, y_pred)
    cm_data_list.append((cm, name))
    report = classification_report(y_val, y_pred, output_dict=True)
    metrics_to_track[name] = {
        "Accuracy": f"{report['accuracy']:.4f}",
        "Precision": f"{report['weighted avg']['precision']:.4f}",
        "Recall": f"{report['weighted avg']['recall']:.4f}",
        "F1-Score": f"{report['weighted avg']['f1-score']:.4f}",
        "ROC AUC": f"{roc_auc_score(y_val, y_prob):.4f}",
        "MCC": f"{matthews_corrcoef(y_val, y_pred):.4f}",
        "Inf Latency (ms)": f"{inf_time * 1000:.4f}",
        "Train Time (s)": f"{train_time:.2f}"
    }

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_data.append((fpr, tpr, name, metrics_to_track[name]["ROC AUC"]))

# -----------------------------------------------------------------------------
# FINAL ALL-MODEL VISUALIZATIONS
# -----------------------------------------------------------------------------
timestamp = int(time.time())

# A. Combined Horizontal Confusion Matrices
if cm_data_list:
    n_models = len(cm_data_list)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5), sharey=True)
    
    if n_models == 1: axes = [axes]

    for i, (ax, (cm, name)) in enumerate(zip(axes, cm_data_list)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Clone", "Clone"])
        disp.plot(cmap=plt.cm.Blues, colorbar=False, ax=ax)
        
        ax.set_title(name, fontsize=11, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        
        if i == 0:
            ax.set_ylabel("True Label", fontsize=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(left=False)
            
    figure_name = selected_lang.upper() if selected_lang != "fastapi" else "JAVA (Type 3/4 Clones)"

    plt.subplots_adjust(wspace=0.1) 
    plt.suptitle(f"Confusion Matrices: {figure_name}", fontsize=14, y=1.05)
    
    plt.savefig(os.path.join(output_dir, f"{selected_lang}_combined_cms_{timestamp}.pdf"), 
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

# B. Summary Table
model_names = list(metrics_to_track.keys())
if model_names:
    headers = ["Metric"] + model_names
    table_data = [[m] + [metrics_to_track[name][m] for name in model_names] 
                  for m in metrics_to_track[model_names[0]].keys()]

    summary_text = f"--- Multi-Model Comparison: {selected_lang.upper()} ---\n"
    summary_text += tabulate(table_data, headers=headers, tablefmt="fancy_grid")
    
    with open(os.path.join(output_dir, f"{selected_lang}_full_comparison_{timestamp}.txt"), "w") as f:
        f.write(summary_text)
    print(summary_text)

    # C. All-Model ROC Plot
    plt.figure(figsize=(10, 8))
    for fpr, tpr, name, auc in roc_data:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title(f"All Models ROC Comparison: {figure_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{selected_lang}_all_roc_{timestamp}.pdf"))
    plt.show()