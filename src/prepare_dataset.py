"""
prepare_dataset.py

Converts the IP102 dataset from its raw format (flat image folder + txt label
files) into the folder structure expected by YOLOv8 classification mode.

Raw format:
    data/raw/ip102_v1.1/images/  <- all images in one flat folder
    data/raw/ip102_v1.1/train.txt <- lines of "filename.jpg class_id"

YOLOv8 format:
    data/processed/train/aphids/00100.jpg
    data/processed/val/aphids/00200.jpg
    data/processed/test/aphids/00300.jpg
"""

import os
import shutil
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/ip102_v1.1")
IMG_DIR    = RAW_DIR / "images"
CLASS_FILE = Path("data/raw/Classification/classes.txt")
OUT_DIR    = Path("data/processed")


def load_classes(class_file):
    """
    Load pest class names from the IP102 classes.txt file.

    Purpose:
        Reads the class index-to-name mapping so images can be copied
        into correctly named folders.

    Args:
        class_file (Path): Path to classes.txt file. Each line is formatted
                           as "1  rice leaf roller", "2  aphids", etc.
                           The file is 1-indexed.

    Returns:
        dict: A dictionary mapping 0-indexed class IDs (int) to cleaned
              class names (str). Spaces and slashes replaced with underscores
              so names are safe to use as folder names.

    Example:
        {0: "rice_leaf_roller", 1: "rice_leaf_caterpillar", ...}
    """
    classes = {}
    with open(class_file) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                idx  = int(parts[0]) - 1
                name = parts[1].strip()
                name = name.replace(" ", "_").replace("/", "_")
                classes[idx] = name
    return classes


def process_split(split_name, img_dir, raw_dir, out_dir, classes):
    """
    Copy all images for one dataset split into YOLOv8 class subfolders.

    Purpose:
        Reads the label file for a given split (train/val/test), looks up
        each image's class, creates the destination folder if needed, and
        copies the image into it. This transforms the flat IP102 layout into
        the nested folder structure YOLOv8 classification mode requires.

    Args:
        split_name (str): One of "train", "val", or "test". Used to find
                          the label file (e.g. train.txt) and name the
                          output folder (e.g. data/processed/train/).
        img_dir    (Path): Path to the folder containing all raw images.
        raw_dir    (Path): Path to the ip102_v1.1 folder containing .txt files.
        out_dir    (Path): Root output folder (data/processed/).
        classes    (dict): Mapping of 0-indexed class ID to class name string,
                           as returned by load_classes().

    Returns:
        tuple: (copied, skipped) where:
            copied  (int) = number of images successfully copied
            skipped (int) = number of images skipped (missing file or unknown class)

    Example:
        copied, skipped = process_split("train", IMG_DIR, RAW_DIR, OUT_DIR, classes)
    """
    txt_file = raw_dir / f"{split_name}.txt"
    copied   = 0
    skipped  = 0

    with open(txt_file) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        filename   = parts[0]
        class_idx  = int(parts[1])
        class_name = classes.get(class_idx)

        if class_name is None:
            skipped += 1
            continue

        src = img_dir / filename
        if not src.exists():
            skipped += 1
            continue

        dst_folder = out_dir / split_name / class_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst_folder / filename)
        copied += 1

    return copied, skipped


def main():
    """
    Main entry point for the dataset preparation pipeline.

    Purpose:
        Orchestrates the full conversion from raw IP102 format to YOLOv8
        classification format. Loads class names, processes all three splits,
        prints a summary, and verifies the output folder structure.

    Args:
        None

    Returns:
        None

    Side effects:
        Creates data/processed/ folder structure with up to 102 subfolders
        per split, each containing the relevant images.
    """
    classes = load_classes(CLASS_FILE)
    print(f"Loaded {len(classes)} classes")

    print("\nFormatting dataset...")
    for split in ["train", "val", "test"]:
        copied, skipped = process_split(split, IMG_DIR, RAW_DIR, OUT_DIR, classes)
        print(f"  {split}: {copied} images copied, {skipped} skipped")

    print("\nDone! Dataset ready at data/processed/")
    print("\nFolder structure sample:")
    for split in ["train", "val", "test"]:
        folders = list((OUT_DIR / split).iterdir())
        print(f"  {split}/  -> {len(folders)} class folders")


if __name__ == "__main__":
    main()