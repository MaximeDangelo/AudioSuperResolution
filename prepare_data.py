"""
Prepare les annotations JSON au format SpeechBrain pour le dataset radio.
Genere train.json, valid.json et test.json (test = copie de valid).
"""
import json
import os
import sys


DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def make_annotation(split, raw_subdir="raw", clean_subdir="clean"):
    """Genere le JSON d'annotations pour un split."""
    raw_dir = os.path.join(DATASET_DIR, split, raw_subdir)
    clean_dir = os.path.join(DATASET_DIR, split, clean_subdir)

    if not os.path.isdir(raw_dir):
        print(f"  SKIP {split}: {raw_dir} n'existe pas")
        return {}

    entries = {}
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".wav"))

    for f in files:
        uid = os.path.splitext(f)[0]
        raw_path = os.path.abspath(os.path.join(raw_dir, f))
        clean_path = os.path.abspath(os.path.join(clean_dir, f))

        if not os.path.exists(clean_path):
            continue

        entries[uid] = {
            "noisy_wav": raw_path,
            "clean_wav": clean_path,
        }

    return entries


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    for split_name, split_dir in [("train", "train"), ("valid", "val"), ("test", "val")]:
        entries = make_annotation(split_dir)
        out_path = os.path.join(DATASET_DIR, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"  {split_name}.json : {len(entries)} paires")


if __name__ == "__main__":
    main()
