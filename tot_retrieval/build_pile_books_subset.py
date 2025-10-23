#!/usr/bin/env python3
"""
Builds a subset of The Pile focused on book and text-heavy sources
for information retrieval or Tip-of-the-Tongue (TOT) experiments.

Downloads (or streams) these subcorpora:
- BookCorpus2
- Books3
- Gutenberg
- Wikipedia
- Pile-CC

Total expected size: ~150–180 GB (depending on subsets kept)
Tested with: Python 3.9+, Ubuntu 20.04+, HF Datasets 2.18+
"""

import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm

# ------------ CONFIGURATION ------------
OUTPUT_DIR = "/data/the_pile_books_subset"  # change this path if needed
SUBSETS = ["BookCorpus2", "Books3", "Gutenberg", "Wikipedia", "Pile-CC"]
STREAMING_MODE = True        # Set False if you want to fully download (needs ~180GB)
SAVE_INTERVAL = 1_000_000    # Save every million records if streaming
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_subset(name):
    """Load one subset of The Pile, streaming if requested."""
    print(f"\n=== Loading subset: {name} ===")
    try:
        ds = load_dataset("EleutherAI/the_pile", name, split="train", streaming=STREAMING_MODE)
        print(f"Loaded subset: {name}")
        return ds
    except Exception as e:
        print(f"[WARN] Failed to load {name}: {e}")
        return None

def save_streaming_dataset(ds, subset_name, out_dir):
    """Iterate through a streaming dataset and write text to disk in chunks."""
    subset_path = os.path.join(out_dir, f"{subset_name}.txt")
    count = 0
    with open(subset_path, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc=f"Saving {subset_name}"):
            text = example.get("text", "")
            if text:
                f.write(text.strip().replace("\n", " ") + "\n")
            count += 1
            if count % SAVE_INTERVAL == 0:
                f.flush()
                print(f"Saved {count:,} examples from {subset_name}...")
    print(f"✅ Finished saving {subset_name}: {count:,} examples to {subset_path}")

def main():
    print("🚀 Starting The Pile book/text subset build...")
    all_datasets = []

    for subset_name in SUBSETS:
        ds = get_subset(subset_name)
        if ds is None:
            continue

        if STREAMING_MODE:
            # Stream and save incrementally
            save_streaming_dataset(ds, subset_name, OUTPUT_DIR)
        else:
            # Fully download and save as HuggingFace dataset
            out_path = os.path.join(OUTPUT_DIR, subset_name)
            print(f"Saving full dataset to {out_path}...")
            ds.save_to_disk(out_path)
            all_datasets.append(ds)

    # If full (non-streaming) mode, merge all
    if not STREAMING_MODE and all_datasets:
        print("Merging all subsets into one HuggingFace dataset...")
        combined = concatenate_datasets(all_datasets)
        combined.save_to_disk(os.path.join(OUTPUT_DIR, "combined_books_subset"))
        print("✅ All subsets merged and saved successfully!")

    print("\n🎉 Done! Your The Pile book subset is ready at:")
    print(f"   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
