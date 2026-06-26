"""
train_binary.py

Trains a YOLOv8 classification model on the binary insect/not_insect
dataset. This model acts as a filtering stage before the 102-class
species classifier - given a cropped trap image, it first decides
whether an insect is present at all.

Usage:
    python src/train_binary.py --epochs 20 --imgsz 224 --model yolov8n-cls.pt
"""

import argparse
from ultralytics import YOLO


def train_binary_model(model_name, data_path, epochs, imgsz, batch):
    """
    Train a YOLOv8 classification model on the binary insect dataset.

    Purpose:
        Fine-tunes a pretrained YOLOv8 classification model on the
        2-class (insect / not_insect) dataset using transfer learning,
        the same approach used for the 102-class species model.

    Args:
        model_name (str): Pretrained YOLOv8 classification model to start
                          from (e.g. "yolov8n-cls.pt").
        data_path  (str): Path to the binary_processed dataset root folder,
                          containing train/val/test subfolders each with
                          insect/ and not_insect/ class folders.
        epochs     (int): Number of training epochs.
        imgsz      (int): Image size to resize inputs to before training.
        batch      (int): Number of images processed together per step.

    Returns:
        ultralytics.engine.model.YOLO: The trained model object.

    Side effects:
        Saves trained weights and training logs under the specified
        project/name folder.
    """
    model = YOLO(model_name)

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=10,
        project="models/checkpoints",
        name="binary_insect_classifier"
    )

    return results


def main():
    """
    Main entry point for binary classifier training.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train binary insect/not_insect classifier")
    parser.add_argument("--model",  type=str, default="yolov8n-cls.pt")
    parser.add_argument("--data",   type=str, default="data/binary_processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz",  type=int, default=224)
    parser.add_argument("--batch",  type=int, default=64)
    args = parser.parse_args()

    print(f"Training {args.model} for {args.epochs} epoch(s) on binary insect dataset...")
    train_binary_model(args.model, args.data, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()