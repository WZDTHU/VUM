"""
Convert ADE20K HuggingFace parquet dataset to standard ADEChallengeData2016 format.

Output structure:
    ADEChallengeData2016/
        images/training/ADE_train_*.jpg
        images/validation/ADE_val_*.jpg
        annotations/training/ADE_train_*.png
        annotations/validation/ADE_val_*.png

The segmentation masks in the parquet are stored as RGB PNGs. For mmseg, we need
single-channel annotation PNGs where each pixel = class_id (0-indexed, with 0 being
the background shifted from the original 150-class scheme).

In ADE20K convention:
- Original annotation has 150 classes (1-150) + 0=unlabeled
- mmseg expects class ids 0-149 + 255=ignore, achieved via reduce_zero_label=True
"""
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


DATA_ROOT = '/path/to/data'
OUTPUT_ROOT = '/path/to/data'


def process_row(row, split):
    filename = row['filename']
    stem = os.path.splitext(filename)[0]

    img_bytes = row['image']['bytes']
    img = Image.open(io.BytesIO(img_bytes))
    img_path = os.path.join(OUTPUT_ROOT, 'images', split, filename)
    img.save(img_path)

    # segmentation mask - first element is the semantic segmentation
    seg_bytes = row['segmentations'][0]['bytes']
    seg_img = Image.open(io.BytesIO(seg_bytes))
    seg_arr = np.array(seg_img)

    # The segmentation PNG from ADE20K is RGB-encoded:
    # pixel value = R/256 * 256 + G/256 (for >150 classes scenario)
    # But for ADE20K challenge (150 classes), it's simply:
    # class_id = R * 256 + G (if 16-bit), or just single value if grayscale
    if len(seg_arr.shape) == 3:
        # RGB encoded: class = (R/256)*256 + G/256 for full ADE20K
        # For 150-class challenge: class_id = R + 1 (R channel has the class)
        seg_gray = seg_arr[:, :, 0]  # R channel contains class id
    else:
        seg_gray = seg_arr

    ann_path = os.path.join(OUTPUT_ROOT, 'annotations', split, stem + '.png')
    ann_img = Image.fromarray(seg_gray.astype(np.uint8))
    ann_img.save(ann_path)
    return filename


def convert_split(split_name, parquet_pattern):
    os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'annotations', split_name), exist_ok=True)

    parquet_files = sorted(Path(DATA_ROOT).glob(parquet_pattern))
    print(f'Found {len(parquet_files)} parquet files for {split_name}')

    count = 0
    for pf in parquet_files:
        print(f'  Processing {pf.name}...')
        df = pd.read_parquet(pf)
        for idx, row in df.iterrows():
            process_row(row, split_name)
            count += 1
            if count % 500 == 0:
                print(f'    Processed {count} images')
    print(f'  Done: {count} images for {split_name}')


if __name__ == '__main__':
    print(f'Converting ADE20K from {DATA_ROOT} to {OUTPUT_ROOT}')
    convert_split('training', 'train-*.parquet')
    convert_split('validation', 'validation-*.parquet')
    print('Conversion complete!')
