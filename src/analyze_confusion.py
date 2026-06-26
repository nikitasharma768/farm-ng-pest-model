"""
analyze_confusion.py

Analyzes the YOLOv8 classification model's predictions on the test set to find
which pest species are most commonly confused with each other. This supports
Prof. Morris's suggestion to check whether top-5 misclassifications are
closely related species, which would justify a weighted top-5 accuracy metric.

Output:
    Prints the top N most confused class pairs, sorted by confusion frequency.
"""

from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import argparse


def get_predictions(model, test_dir):
    """
    Run the model on every image in the test set and collect true vs
    predicted labels.

    Purpose:
        Walks through each class folder in the test set, runs the model on
        every image, and records the true class alongside the model's top-5
        predicted classes. This raw prediction data is needed to compute
        which classes get confused with which others.

    Args:
        model    (YOLO): A loaded YOLOv8 classification model.
        test_dir (Path): Path to data/processed/test/, containing one
                         subfolder per class with images inside.

    Returns:
        list: A list of dicts, one per image, each containing:
              {"true": str, "top1": str, "top5": list[str]}
    """
    results = []
    class_folders = sorted([f for f in test_dir.iterdir() if f.is_dir()])

    for class_folder in class_folders:
        true_label = class_folder.name
        images = list(class_folder.glob("*.jpg"))

        for img_path in images:
            pred = model.predict(str(img_path), verbose=False)[0]
            top5_idx = pred.probs.top5
            top5_names = [model.names[i] for i in top5_idx]

            results.append({
                "true": true_label,
                "top1": top5_names[0],
                "top5": top5_names
            })

    return results


def find_confused_pairs(results, top_n=20):
    """
    Count how often each (true, predicted) class pair occurs when the
    model's top-1 prediction is wrong.

    Purpose:
        Identifies which specific species pairs the model mixes up most
        often. This is the core analysis Prof. Morris asked for - checking
        whether misclassifications cluster around related species.

    Args:
        results (list): Output from get_predictions().
        top_n   (int):  How many top confused pairs to return.

    Returns:
        list: A list of tuples (true_label, predicted_label, count),
              sorted by count descending, limited to top_n entries.
    """
    confusion_counts = defaultdict(int)

    for r in results:
        if r["true"] != r["top1"]:
            pair = (r["true"], r["top1"])
            confusion_counts[pair] += 1

    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: -x[1])
    return [(pair[0], pair[1], count) for pair, count in sorted_pairs[:top_n]]


def top5_contains_related(results):
    """
    Check, for images the model got wrong in top-1, how often the correct
    answer still appears somewhere in the top-5.

    Purpose:
        Measures how often a "miss" is actually a near-miss - the model
        ranked the correct species 2nd through 5th rather than completely
        unrelated. High recovery in top-5 supports the idea of a weighted
        scoring system.

    Args:
        results (list): Output from get_predictions().

    Returns:
        dict: {
            "total_wrong_top1": int,
            "recovered_in_top5": int,
            "recovery_rate": float
        }
    """
    wrong_top1 = [r for r in results if r["true"] != r["top1"]]
    recovered = [r for r in wrong_top1 if r["true"] in r["top5"]]

    return {
        "total_wrong_top1": len(wrong_top1),
        "recovered_in_top5": len(recovered),
        "recovery_rate": len(recovered) / len(wrong_top1) if wrong_top1 else 0
    }


def main():
    """
    Main entry point. Loads the model, runs predictions on the test set,
    and prints the most confused class pairs plus top-5 recovery stats.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Analyze model confusion patterns")
    parser.add_argument("--model", type=str, default="models/checkpoints/best.pt")
    parser.add_argument("--test_dir", type=str, default="data/processed/test")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--sample", type=int, default=None,
                         help="If set, only test this many images per class (faster)")
    args = parser.parse_args()

    model = YOLO(args.model)
    test_dir = Path(args.test_dir)

    print("Running predictions on test set... this may take a while on CPU.")
    results = get_predictions(model, test_dir)

    print(f"\nTotal test images evaluated: {len(results)}")

    print(f"\nTop {args.top_n} most confused species pairs (true -> predicted):")
    pairs = find_confused_pairs(results, args.top_n)
    for true_label, pred_label, count in pairs:
        print(f"  {true_label:35s} -> {pred_label:35s}  ({count} times)")

    print("\nTop-5 recovery analysis:")
    stats = top5_contains_related(results)
    print(f"  Wrong in top-1       : {stats['total_wrong_top1']}")
    print(f"  Correct in top-5     : {stats['recovered_in_top5']}")
    print(f"  Recovery rate        : {stats['recovery_rate']:.1%}")


if __name__ == "__main__":
    main()