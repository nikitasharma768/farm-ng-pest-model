"""
evaluate_binary.py

Evaluates the trained binary insect/not_insect classifier using F1 score,
precision, and recall - not just accuracy - since the test set is
intentionally imbalanced (more insect images than not_insect images).
F1 score is more reliable than accuracy on imbalanced data because it
accounts for both false positives and false negatives directly.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, test_dir):
    """
    Run the binary classifier on every test image and compute precision,
    recall, and F1 score for the "insect" class.

    Args:
        model_path (str): Path to the trained binary classifier weights.
        test_dir   (Path): Path to data/binary_processed/test/, containing
                           insect/ and not_insect/ subfolders.

    Returns:
        dict: counts and metrics (precision, recall, f1, accuracy).
    """
    model = YOLO(model_path)

    tp = fp = tn = fn = 0

    for class_name in ["insect", "not_insect"]:
        class_dir = test_dir / class_name
        images = list(class_dir.glob("*.jpg"))

        for img_path in images:
            result = model.predict(str(img_path), verbose=False)[0]
            pred_idx = result.probs.top1
            pred_label = result.names[pred_idx]

            if class_name == "insect":
                if pred_label == "insect":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_label == "not_insect":
                    tn += 1
                else:
                    fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy
    }


def main():
    """
    Main entry point. Loads the trained model, evaluates it on the test
    set, and prints precision, recall, F1, and accuracy.
    """
    parser = argparse.ArgumentParser(description="Evaluate binary insect classifier with F1 score")
    parser.add_argument("--model", type=str, default="models/checkpoints/binary_insect_classifier/weights/best.pt")
    parser.add_argument("--test_dir", type=str, default="data/binary_processed/test")
    args = parser.parse_args()

    metrics = evaluate_model(args.model, Path(args.test_dir))

    print("\nConfusion counts:")
    print(f"  True Positives  (insect correctly identified)     : {metrics['true_positives']}")
    print(f"  False Positives (not_insect predicted as insect)  : {metrics['false_positives']}")
    print(f"  True Negatives  (not_insect correctly identified) : {metrics['true_negatives']}")
    print(f"  False Negatives (insect predicted as not_insect)  : {metrics['false_negatives']}")

    print("\nMetrics:")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1 Score  : {metrics['f1']:.3f}")
    print(f"  Accuracy  : {metrics['accuracy']:.3f}  (less reliable here due to class imbalance)")


if __name__ == "__main__":
    main()