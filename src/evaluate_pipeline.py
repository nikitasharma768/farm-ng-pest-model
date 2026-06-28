"""
evaluate_pipeline.py

Evaluates the full two-stage pipeline (binary filter -> species classifier)
end to end, rather than evaluating each model in isolation. This answers
the practical question: "if I feed real trap images through the complete
system, how often does it correctly identify the species?"

This differs from the standalone model evaluations:
  - evaluate_binary.py measures Stage A alone (insect vs not_insect)
  - The species model's own validation measures Stage B alone, assuming
    every image it sees actually contains an insect
  - This script measures the realistic combined outcome, including cases
    where Stage A incorrectly blocks a real insect before Stage B ever
    gets a chance to classify it
"""

import argparse
from pathlib import Path
from pipeline import PestPipeline


def evaluate_on_species_test_set(pipeline, test_dir):
    """
    Run the full pipeline on the IP102 species test set and measure how
    often the correct species comes out the other end.

    Purpose:
        Every image in this test set is a known, real insect with a known
        species label. Running it through the full pipeline (not just the
        species model alone) reveals how often Stage A's imperfect filtering
        costs accuracy - if Stage A wrongly says "not insect," the image
        never reaches species classification at all, which counts as a
        miss for the combined system even though the species model itself
        was never given the chance to be right or wrong.

    Args:
        pipeline (PestPipeline): A loaded two-stage pipeline instance.
        test_dir (Path): Path to data/processed/test/, containing one
                         subfolder per species with images inside.

    Returns:
        dict: {
            "total": int,
            "blocked_by_stage_a": int,      # real insect, but Stage A said not_insect
            "correct_species": int,         # passed Stage A and got the right species
            "wrong_species": int,           # passed Stage A but wrong species
            "pipeline_top1_acc": float,     # correct_species / total
            "stage_a_pass_rate": float      # (correct_species + wrong_species) / total
        }
    """
    total = 0
    blocked = 0
    correct = 0
    wrong = 0

    species_folders = sorted([f for f in test_dir.iterdir() if f.is_dir()])

    for species_folder in species_folders:
        true_label = species_folder.name
        images = list(species_folder.glob("*.jpg"))

        for img_path in images:
            total += 1
            result = pipeline.classify_image(img_path)

            if not result["is_insect"]:
                blocked += 1
                continue

            if result["species"] == true_label:
                correct += 1
            else:
                wrong += 1

    pipeline_top1_acc = correct / total if total > 0 else 0
    stage_a_pass_rate = (correct + wrong) / total if total > 0 else 0

    return {
        "total": total,
        "blocked_by_stage_a": blocked,
        "correct_species": correct,
        "wrong_species": wrong,
        "pipeline_top1_acc": pipeline_top1_acc,
        "stage_a_pass_rate": stage_a_pass_rate
    }


def main():
    """
    Main entry point. Loads both models, runs the combined pipeline
    evaluation, and prints results.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Evaluate the full two-stage pipeline")
    parser.add_argument("--binary_model", type=str,
                         default="models/checkpoints/binary_insect_classifier/weights/best.pt")
    parser.add_argument("--species_model", type=str,
                         default="models/checkpoints/best.pt")
    parser.add_argument("--test_dir", type=str, default="data/processed/test")
    parser.add_argument("--sample_per_class", type=int, default=None,
                         help="If set, only test this many images per species (faster)")
    args = parser.parse_args()

    pipeline = PestPipeline(args.binary_model, args.species_model)

    print("\nRunning full pipeline evaluation on species test set...")
    print("(This measures the combined system, not just one model in isolation)\n")

    metrics = evaluate_on_species_test_set(pipeline, Path(args.test_dir))

    print(f"Total real-insect test images : {metrics['total']}")
    print(f"Blocked by Stage A (binary)    : {metrics['blocked_by_stage_a']}  "
          f"({metrics['blocked_by_stage_a']/metrics['total']:.1%})")
    print(f"Passed Stage A, correct species: {metrics['correct_species']}  "
          f"({metrics['correct_species']/metrics['total']:.1%})")
    print(f"Passed Stage A, wrong species  : {metrics['wrong_species']}  "
          f"({metrics['wrong_species']/metrics['total']:.1%})")
    print(f"\nStage A pass-through rate      : {metrics['stage_a_pass_rate']:.1%}")
    print(f"Combined pipeline top-1 accuracy: {metrics['pipeline_top1_acc']:.1%}")


if __name__ == "__main__":
    main()