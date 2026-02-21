"""
Calculate average cosine distances for embedding pairs to assess 
how well models separate clones (Label 1) from non-clones (Label 0).
"""

import json
import torch
import torch.nn.functional as F

BASE_PATH = "../clone_detector/data/embeddings"

DATASETS = {
    "Java": {
        "Java CodeT5+ (base)": f"{BASE_PATH}/CodeT5P/java_codet5p_embeddings_base.jsonl",
        "Java CodeT5+ (tuned)": f"{BASE_PATH}/CodeT5P/java_codet5p_embeddings_tuned.jsonl",
        "Java GraphCodeBERT (base)": f"{BASE_PATH}/GraphCodeBERT/java_graphcodebert_embeddings_base.jsonl",
        "Java GraphCodeBERT (tuned, mean)": f"{BASE_PATH}/GraphCodeBERT/java_graphcodebert_embeddings_tuned_mean.jsonl",
        "Java GraphCodeBERT (tuned, CLS)": f"{BASE_PATH}/GraphCodeBERT/java_graphcodebert_embeddings_tuned_cls.jsonl",
    },
    "Python": {
        "Python CodeT5+ (base)": f"{BASE_PATH}/CodeT5P/python_codet5p_embeddings_base.jsonl",
        "Python CodeT5+ (tuned)": f"{BASE_PATH}/CodeT5P/python_codet5p_embeddings_tuned.jsonl",
        "Python GraphCodeBERT (base)": f"{BASE_PATH}/GraphCodeBERT/python_graphcodebert_embeddings_base.jsonl",
        "Python GraphCodeBERT (tuned, mean)": f"{BASE_PATH}/GraphCodeBERT/python_graphcodebert_embeddings_tuned_mean.jsonl",
        "Python GraphCodeBERT (tuned, CLS)": f"{BASE_PATH}/GraphCodeBERT/python_graphcodebert_embeddings_tuned_cls.jsonl",
    },
}

def cosine_distance(a, b):
    return 1 - F.cosine_similarity(a, b, dim=0)


def compute_avg_distances(path):
    label_distances = {0: [], 1: []}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            emb1 = torch.tensor(obj["embedding1"], dtype=torch.float)
            emb2 = torch.tensor(obj["embedding2"], dtype=torch.float)
            label = obj["label"]

            dist = cosine_distance(emb1, emb2).item()
            label_distances[label].append(dist)

    return {
        label: sum(dists) / len(dists)
        for label, dists in label_distances.items()
        if len(dists) > 0
    }


def select_language():
    languages = list(DATASETS.keys())
    print("Select language:")
    for i, lang in enumerate(languages, 1):
        print(f"{i}. {lang}")

    idx = int(input("Enter number: ")) - 1
    return languages[idx]


def print_table(language):
    print(f"\nAverage cosine distances for {language} datasets\n")
    header = f"{'Model':45} {'Label 0':>12} {'Label 1':>12}"
    print(header)
    print("-" * len(header))

    for name, path in DATASETS[language].items():
        avg = compute_avg_distances(path)
        l0 = f"{avg.get(0, float('nan')):.6f}"
        l1 = f"{avg.get(1, float('nan')):.6f}"
        print(f"{name:45} {l0:>12} {l1:>12}")


if __name__ == "__main__":
    language = select_language()
    print_table(language)