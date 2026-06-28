"""
pipeline.py

End-to-end inference pipeline combining the two trained classifiers:
  Stage A: binary insect/not_insect filter
  Stage B: 102-species classifier (only runs if Stage A says "insect")

This mirrors the intended Amiga deployment logic: a cropped trap image
first gets checked for "is there even an insect here," and only insect-
positive images get passed on to species identification. This avoids
wasting species-classification effort (and avoiding noisy low-confidence
guesses) on images with no insect present.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


class PestPipeline:
    """
    Two-stage inference pipeline: binary insect filter -> species classifier.
    """

    def __init__(self, binary_model_path, species_model_path, insect_threshold=0.5):
        """
        Load both trained models once, so they aren't reloaded per image.

        Purpose:
            Loading a YOLO model from disk is relatively slow. For batch
            processing (or eventually real-time use on the Amiga), both
            models should be loaded once at startup and reused for every
            image, rather than reloaded each time.

        Args:
            binary_model_path  (str): Path to the trained binary classifier
                                      weights (insect/not_insect).
            species_model_path (str): Path to the trained 102-species
                                      classifier weights.
            insect_threshold    (float): Minimum confidence required for
                                      Stage A to classify an image as
                                      "insect" before passing it to Stage B.
                                      Defaults to 0.5 (whichever class wins).

        Returns:
            None
        """
        print("Loading binary classifier...")
        self.binary_model = YOLO(binary_model_path)
        print("Loading species classifier...")
        self.species_model = YOLO(species_model_path)
        self.insect_threshold = insect_threshold

    def classify_image(self, image_path):
        """
        Run the full two-stage pipeline on a single image.

        Purpose:
            Implements the core pipeline logic: check if an insect is
            present first; only run species classification if it is.
            This is the function that would be called per cropped trap
            image once Stage 1 (trap detection) is built.

        Args:
            image_path (str): Path to a single image file.

        Returns:
            dict: {
                "image": str,
                "is_insect": bool,
                "insect_confidence": float,
                "species": str or None,        # None if not_insect
                "species_confidence": float or None,
                "stage_reached": str            # "binary_only" or "full_pipeline"
            }
        """
        binary_result = self.binary_model.predict(str(image_path), verbose=False)[0]
        pred_idx = binary_result.probs.top1
        pred_label = binary_result.names[pred_idx]
        pred_conf = float(binary_result.probs.top1conf)

        is_insect = (pred_label == "insect") and (pred_conf >= self.insect_threshold)

        result = {
            "image": str(image_path),
            "is_insect": is_insect,
            "insect_confidence": round(pred_conf, 4),
            "species": None,
            "species_confidence": None,
            "stage_reached": "binary_only"
        }

        if not is_insect:
            return result

        species_result = self.species_model.predict(str(image_path), verbose=False)[0]
        species_idx = species_result.probs.top1
        species_label = species_result.names[species_idx]
        species_conf = float(species_result.probs.top1conf)

        result["species"] = species_label
        result["species_confidence"] = round(species_conf, 4)
        result["stage_reached"] = "full_pipeline"

        return result

    def classify_batch(self, image_paths):
        """
        Run the pipeline on a list of images and collect results.

        Purpose:
            Convenience wrapper for processing many images at once
            (e.g. a folder of trap crops), used for testing and reporting.

        Args:
            image_paths (list): List of image file paths (str or Path).

        Returns:
            list: List of result dicts, one per image, as returned by
                  classify_image().
        """
        results = []
        for img_path in image_paths:
            results.append(self.classify_image(img_path))
        return results


def main():
    """
    Main entry point. Runs the pipeline on a single image or a folder of
    images and prints a summary.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run the two-stage pest detection pipeline")
    parser.add_argument("--binary_model", type=str,
                         default="models/checkpoints/binary_insect_classifier/weights/best.pt")
    parser.add_argument("--species_model", type=str,
                         default="models/checkpoints/best.pt")
    parser.add_argument("--input", type=str, required=True,
                         help="Path to a single image or a folder of images")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Minimum confidence to classify as insect")
    args = parser.parse_args()

    pipeline = PestPipeline(args.binary_model, args.species_model, args.threshold)

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob("*.jpg"))

    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    results = pipeline.classify_batch(image_paths)

    for r in results:
        if r["is_insect"]:
            print(f"  {Path(r['image']).name:30s} -> INSECT (conf {r['insect_confidence']:.2f}) "
                  f"-> {r['species']} (conf {r['species_confidence']:.2f})")
        else:
            print(f"  {Path(r['image']).name:30s} -> NOT INSECT (conf {r['insect_confidence']:.2f})")

    n_insect = sum(1 for r in results if r["is_insect"])
    print(f"\nSummary: {n_insect}/{len(results)} classified as insect and passed to species model")


if __name__ == "__main__":
    main()