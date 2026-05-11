"""
Convert HuggingFace COCO parquet to standard COCO directory structure.
Extracts images from parquet and downloads official annotation JSONs.

Usage:
    source activate py12cu128torch290
    python projects/evaluate/det/convert_coco.py

Output structure:
    data/coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   ├── 000000000009.jpg
    │   └── ...
    └── val2017/
        ├── 000000000139.jpg
        └── ...
"""

import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

PARQUET_ROOT = "/path/to/data"
OUTPUT_ROOT = "/path/to/data"

ANNOTATION_URLS = {
    "annotations/instances_train2017.json": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def extract_images_from_shard(parquet_path, output_dir):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    count = 0
    for _, row in df.iterrows():
        image_id = int(row["image_id"])
        image_bytes = row["image"]["bytes"]
        filename = f"{image_id:012d}.jpg"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            count += 1
    return os.path.basename(parquet_path), count, len(df)


def main():
    os.makedirs(os.path.join(OUTPUT_ROOT, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "annotations"), exist_ok=True)

    # Step 1: Download annotations
    ann_zip = os.path.join(OUTPUT_ROOT, "annotations_trainval2017.zip")
    ann_file = os.path.join(OUTPUT_ROOT, "annotations", "instances_train2017.json")
    if not os.path.exists(ann_file):
        print("Downloading COCO annotations...")
        subprocess.run([
            "wget", "-c", "-q", "--show-progress",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "-O", ann_zip
        ], check=True)
        print("Extracting annotations...")
        subprocess.run(["unzip", "-o", "-q", ann_zip, "-d", OUTPUT_ROOT], check=True)
        os.remove(ann_zip)
        print("Annotations ready.")
    else:
        print("Annotations already exist, skipping download.")

    # Step 2: Extract images from parquet shards
    for split, prefix in [("val2017", "val-"), ("train2017", "train-")]:
        output_dir = os.path.join(OUTPUT_ROOT, split)
        shards = sorted(glob.glob(os.path.join(PARQUET_ROOT, f"{prefix}*.parquet")))
        print(f"\nExtracting {split}: {len(shards)} shards → {output_dir}")

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(extract_images_from_shard, s, output_dir): s
                for s in shards
            }
            total_new, total_all = 0, 0
            for future in as_completed(futures):
                name, new, total = future.result()
                total_new += new
                total_all += total
                print(f"  {name}: {new} new / {total} total")
        print(f"  {split} done: {total_new} new images extracted, {total_all} total in parquet")

    # Verify
    for split in ["train2017", "val2017"]:
        n = len(os.listdir(os.path.join(OUTPUT_ROOT, split)))
        print(f"{split}: {n} images")
    for ann in ["instances_train2017.json", "instances_val2017.json"]:
        p = os.path.join(OUTPUT_ROOT, "annotations", ann)
        print(f"{ann}: {'EXISTS' if os.path.exists(p) else 'MISSING'}")


if __name__ == "__main__":
    main()
