"""
make_subset.py

Creates a tiny subset of the processed dataset for quick local testing.
This lets us verify the training pipeline works correctly on CPU before
running the full training job on Google Colab with a GPU.

Output:
    data/processed_subset/train/<class>/  (a few images per class)
    data/processed_subset/val/<class>/
"""

import shutil
import random
from pathlib import Path


def create_subset(source_dir, dest_dir, num_classes, images_per_class):
    """
    Copy a small number of classes and images into a subset folder.

    Purpose:
        Builds a tiny version of the dataset so training can be tested
        quickly on CPU, without waiting hours for the full 102-class,
        75,000-image dataset to process.

    Args:
        source_dir        (Path): Path to the full processed dataset
                                  (e.g. data/processed/train).
        dest_dir           (Path): Path to write the subset
                                  (e.g. data/processed_subset/train).
        num_classes        (int): How many class folders to include.
        images_per_class   (int): How many images to copy per class.

    Returns:
        int: Total number of images copied.

    Side effects:
        Creates dest_dir and copies selected images into it.
    """
    class_folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
    selected_classes = class_folders[:num_classes]

    total_copied = 0
    for class_folder in selected_classes:
        images = list(class_folder.glob("*.jpg"))
        random.shuffle(images)
        selected_images = images[:images_per_class]

        dest_class_folder = dest_dir / class_folder.name
        dest_class_folder.mkdir(parents=True, exist_ok=True)

        for img in selected_images:
            shutil.copy2(img, dest_class_folder / img.name)
            total_copied += 1

    return total_copied


def main():
    """
    Main entry point for subset creation.

    Purpose:
        Creates a small train and val subset using the first 3 classes
        and 20 images per class, for quick pipeline testing.

    Args:
        None

    Returns:
        None
    """
    NUM_CLASSES = 3
    IMAGES_PER_CLASS_TRAIN = 20
    IMAGES_PER_CLASS_VAL = 5

    base = Path("data/processed")
    subset_base = Path("data/processed_subset")

    train_copied = create_subset(
        base / "train", subset_base / "train",
        NUM_CLASSES, IMAGES_PER_CLASS_TRAIN
    )
    val_copied = create_subset(
        base / "val", subset_base / "val",
        NUM_CLASSES, IMAGES_PER_CLASS_VAL
    )

    print(f"Subset created at {subset_base}")
    print(f"  Train: {train_copied} images across {NUM_CLASSES} classes")
    print(f"  Val:   {val_copied} images across {NUM_CLASSES} classes")


if __name__ == "__main__":
    main()