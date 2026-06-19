"""
train.py

Trains a YOLOv8 classification model on the IP102 pest dataset.

This script can be run two ways:
  1. Locally with a small subset (epochs=1, tiny test) to verify the
     pipeline works correctly before using a GPU.
  2. On Google Colab with the full dataset and more epochs, using a GPU.

Usage:
    python src/train.py --epochs 1 --imgsz 224 --model yolov8n-cls.pt
"""

import argparse
from ultralytics import YOLO


def train_model(model_name, data_path, epochs, imgsz, batch):
    """
    Train a YOLOv8 classification model on the prepared pest dataset.

    Purpose:
        Loads a pretrained YOLOv8 classification model and fine-tunes it
        on our 102-class pest dataset using transfer learning.

    Args:
        model_name (str): Pretrained YOLOv8 classification model to start
                          from. Options: "yolov8n-cls.pt" (nano, fastest),
                          "yolov8s-cls.pt" (small, more accurate).
        data_path  (str): Path to the data.yaml file or the processed
                          dataset root folder. YOLOv8 classification mode
                          accepts a folder path directly.
        epochs     (int): Number of training epochs (full passes through
                          the training data).
        imgsz      (int): Image size to resize inputs to before training.
                          Smaller = faster but less detail (e.g. 224).
        batch      (int): Number of images processed together per training
                          step. Smaller batch = less memory needed.

    Returns:
        ultralytics.engine.model.YOLO: The trained model object, which
        also contains training metrics and the path to saved weights.

    Side effects:
        Saves trained weights and training logs under runs/classify/.
    """
    model = YOLO(model_name)

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=10,
        project="models/checkpoints",
        name="pest_classifier"
    )

    return results


def main():
    """
    Main entry point for training. Parses command line arguments and
    starts the training run.

    Purpose:
        Allows running training with different settings from the command
        line without editing the script (e.g. quick local test vs full
        Colab training).

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train YOLOv8 pest classifier")
    parser.add_argument("--model",  type=str, default="yolov8n-cls.pt",
                         help="Pretrained model to fine-tune from")
    parser.add_argument("--data",   type=str, default="data/processed",
                         help="Path to processed dataset folder")
    parser.add_argument("--epochs", type=int, default=1,
                         help="Number of training epochs")
    parser.add_argument("--imgsz",  type=int, default=224,
                         help="Image size for training")
    parser.add_argument("--batch",  type=int, default=16,
                         help="Batch size")
    args = parser.parse_args()

    print(f"Training {args.model} for {args.epochs} epoch(s)...")
    print(f"Data: {args.data}")

    train_model(args.model, args.data, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()