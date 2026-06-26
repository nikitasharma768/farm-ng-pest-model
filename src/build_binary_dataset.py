"""
build_binary_dataset.py

Builds a binary classification dataset (insect vs. not-insect) for the
filtering stage Prof. Morris suggested. Combines:
  - Positive class ("insect"): IP102 images, ignoring species label
  - Negative class ("not_insect"): timm/mini-imagenet images, with the
    3 confirmed insect classes (ladybug, rhinoceros_beetle, ant) excluded

mini-imagenet does not contain enough non-insect images to match IP102's
full size, so the test split is balanced by randomly sampling an equal
number of insect images rather than using all 22,619 - this keeps the
test set balanced, which matters for a binary classifier's accuracy
metric to be meaningful.

Output structure (YOLOv8 classification format):
    data/binary_processed/
    ├── train/
    │   ├── insect/
    │   └── not_insect/
    ├── val/
    │   ├── insect/
    │   └── not_insect/
    └── test/
        ├── insect/
        └── not_insect/
"""

import shutil
import random
from pathlib import Path
from datasets import load_dataset

EXCLUDED_INSECT_WNIDS = {
    "n02165456",  # ladybug
    "n02174001",  # rhinoceros_beetle
    "n02219486",  # ant
}

IP102_TRAIN_DIR = Path("data/processed/train")
IP102_VAL_DIR = Path("data/processed/val")
IP102_TEST_DIR = Path("data/processed/test")
OUT_DIR = Path("data/binary_processed")

random.seed(42)  # reproducible sampling


def collect_ip102_images(split_dir):
    """
    Collect all image file paths from one IP102 split, across all species.

    Args:
        split_dir (Path): Path to an IP102 split folder, containing one
                          subfolder per species.

    Returns:
        list: List of Path objects pointing to every image in the split.
    """
    images = []
    for species_folder in split_dir.iterdir():
        if species_folder.is_dir():
            images.extend(species_folder.glob("*.jpg"))
    return images


def copy_insect_images(image_paths, dest_dir, max_images=None):
    """
    Copy IP102 images into the binary dataset's "insect" folder.

    Purpose:
        Builds the positive class. If max_images is set and fewer than
        the full available count, images are randomly sampled so the
        resulting class size matches the negative class for that split.

    Args:
        image_paths (list): List of Path objects to copy.
        dest_dir    (Path): Destination folder.
        max_images  (int or None): If set, randomly sample this many images
                                   instead of copying everything.

    Returns:
        int: Number of images copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if max_images is not None and max_images < len(image_paths):
        image_paths = random.sample(image_paths, max_images)

    count = 0
    for img_path in image_paths:
        new_name = f"{img_path.parent.name}_{img_path.name}"
        shutil.copy2(img_path, dest_dir / new_name)
        count += 1
    return count


def build_not_insect_split(hf_split_name, dest_dir, wnid_to_name, max_images=None, skip_first=0):
    """
    Build the "not_insect" class for one split using mini-imagenet.

    Args:
        hf_split_name (str): Which mini-imagenet split to pull from.
        dest_dir      (Path): Destination folder.
        wnid_to_name  (list): Ordered list mapping label index to WNID.
        max_images    (int or None): Optional cap on number of images saved.
        skip_first    (int): Number of valid images to skip before saving,
                             to avoid reusing images from another split.

    Returns:
        int: Number of images saved.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("timm/mini-imagenet", split=hf_split_name)

    saved = 0
    valid_seen = 0
    for i, example in enumerate(dataset):
        if max_images is not None and saved >= max_images:
            break

        label_idx = example["label"]
        wnid = wnid_to_name[label_idx]
        if wnid in EXCLUDED_INSECT_WNIDS:
            continue

        valid_seen += 1
        if valid_seen <= skip_first:
            continue

        image = example["image"].convert("RGB")
        image.save(dest_dir / f"{hf_split_name}_{i}.jpg")
        saved += 1

    return saved


def main():
    """
    Main entry point. Builds the full binary insect/not_insect dataset,
    balancing the test split by sampling insect images down to match the
    maximum number of not_insect images actually available.

    Args:
        None

    Returns:
        None
    """
    print("Building 'insect' class (train/val) from IP102...")
    train_imgs = collect_ip102_images(IP102_TRAIN_DIR)
    val_imgs = collect_ip102_images(IP102_VAL_DIR)
    test_imgs = collect_ip102_images(IP102_TEST_DIR)

    n_train = copy_insect_images(train_imgs, OUT_DIR / "train" / "insect")
    n_val = copy_insect_images(val_imgs, OUT_DIR / "val" / "insect")

    print(f"  train/insect: {n_train}")
    print(f"  val/insect:   {n_val}")

    print("\nBuilding 'not_insect' class from mini-imagenet (train/val)...")
    dataset_for_names = load_dataset("timm/mini-imagenet", split="train")
    wnid_to_name = dataset_for_names.features["label"].names

    n_train_neg = build_not_insect_split(
        "train", OUT_DIR / "train" / "not_insect", wnid_to_name, max_images=n_train
    )
    n_val_neg = build_not_insect_split(
        "validation", OUT_DIR / "val" / "not_insect", wnid_to_name, max_images=n_val
    )

    print(f"  train/not_insect: {n_train_neg}")
    print(f"  val/not_insect:   {n_val_neg}")

    print("\nBuilding 'not_insect' for test split (mini-imagenet test, then train leftovers)...")
    # Use mini-imagenet's own test split first
    n_test_neg = build_not_insect_split(
        "test", OUT_DIR / "test" / "not_insect", wnid_to_name, max_images=None
    )
    # Top up with leftover train images not already used for train/not_insect
    n_topup = build_not_insect_split(
        "train", OUT_DIR / "test" / "not_insect", wnid_to_name,
        max_images=None, skip_first=n_train_neg
    )
    n_test_neg += n_topup
    print(f"  test/not_insect available: {n_test_neg}")

    print(f"\nBalancing test split: sampling {n_test_neg} insect images to match...")
    n_test = copy_insect_images(test_imgs, OUT_DIR / "test" / "insect", max_images=n_test_neg)
    print(f"  test/insect:      {n_test}")
    print(f"  test/not_insect:  {n_test_neg}")

    print("\nDone! Balanced binary dataset ready at data/binary_processed/")


if __name__ == "__main__":
    main()