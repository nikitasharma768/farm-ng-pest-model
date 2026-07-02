"""
pipeline.py

End-to-end inference pipeline combining all three stages:
  Stage 0: Yellow trap detector (image processing)
  Stage A: Binary insect/not_insect filter (YOLOv8)
  Stage B: 102-species classifier (YOLOv8)

This is the complete system as it would run on the farm-ng Amiga:
  1. Camera captures a frame
  2. Stage 0 finds and crops the yellow sticky trap
  3. Stage A checks if an insect is present on the trap
  4. Stage B identifies the pest species (only if Stage A says insect)
  5. Result is logged with species, confidence, and location

The pipeline also works without trap detection (direct scan mode):
  pass --skip_trap_detection to run Stage A and B directly on the image.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from trap_detector import detect_trap


class PestPipeline:
    """
    Full three-stage inference pipeline: trap detector -> binary filter
    -> species classifier.
    """

    def __init__(self, binary_model_path, species_model_path,
                 insect_threshold=0.5, skip_trap_detection=False):
        """
        Load both trained models once at startup.

        Purpose:
            Loading YOLO models from disk is slow. Both models are loaded
            once here and reused for every image in a batch, rather than
            reloading per image.

        Args:
            binary_model_path    (str): Path to binary insect classifier weights.
            species_model_path   (str): Path to 102-species classifier weights.
            insect_threshold   (float): Minimum confidence for Stage A to pass
                                        an image as "insect" to Stage B.
            skip_trap_detection (bool): If True, skip Stage 0 and run Stage A/B
                                        directly on the input image (direct scan
                                        mode, no trap required).

        Returns:
            None
        """
        print("Loading binary classifier...")
        self.binary_model = YOLO(binary_model_path)
        print("Loading species classifier...")
        self.species_model = YOLO(species_model_path)
        self.insect_threshold = insect_threshold
        self.skip_trap_detection = skip_trap_detection

    def classify_image(self, image_path):
        """
        Run the full three-stage pipeline on a single image.

        Purpose:
            Stage 0 finds and crops the trap region. Stage A checks for
            insect presence. Stage B identifies species. If Stage 0 fails
            to find a trap, the image is skipped unless skip_trap_detection
            is True, in which case Stage A/B run on the full image directly.

        Args:
            image_path (str or Path): Path to a single image file.

        Returns:
            dict: {
                "image"              : str,
                "trap_detected"      : bool,
                "trap_bbox"          : tuple or None,
                "is_insect"          : bool,
                "insect_confidence"  : float or None,
                "species"            : str or None,
                "species_confidence" : float or None,
                "stage_reached"      : str
            }
        """
        result = {
            "image": str(image_path),
            "trap_detected": False,
            "trap_bbox": None,
            "is_insect": False,
            "insect_confidence": None,
            "species": None,
            "species_confidence": None,
            "stage_reached": "none"
        }

        # Stage 0: trap detection
        if not self.skip_trap_detection:
            cropped, bbox = detect_trap(image_path)
            if cropped is None:
                result["stage_reached"] = "no_trap_found"
                return result
            result["trap_detected"] = True
            result["trap_bbox"] = bbox

            # Save cropped trap to a temp file for YOLO to read
            temp_path = Path("data/trap_crops/temp_crop.jpg")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(temp_path), cropped)
            classify_path = temp_path
        else:
            classify_path = image_path
            result["trap_detected"] = True
            result["stage_reached"] = "skip_trap_detection"

        # Stage A: binary insect filter
        binary_result = self.binary_model.predict(
            str(classify_path), verbose=False)[0]
        pred_idx = binary_result.probs.top1
        pred_label = binary_result.names[pred_idx]
        pred_conf = float(binary_result.probs.top1conf)

        result["insect_confidence"] = round(pred_conf, 4)
        is_insect = (pred_label == "insect") and (pred_conf >= self.insect_threshold)
        result["is_insect"] = is_insect
        result["stage_reached"] = "binary_only"

        if not is_insect:
            return result

        # Stage B: species classifier
        species_result = self.species_model.predict(
            str(classify_path), verbose=False)[0]
        species_idx = species_result.probs.top1
        species_label = species_result.names[species_idx]
        species_conf = float(species_result.probs.top1conf)

        result["species"] = species_label
        result["species_confidence"] = round(species_conf, 4)
        result["stage_reached"] = "full_pipeline"

        return result

    def classify_batch(self, image_paths):
        """
        Run the full pipeline on a list of images.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            list: List of result dicts, one per image.
        """
        results = []
        for img_path in image_paths:
            results.append(self.classify_image(img_path))
        return results


def main():
    """
    Main entry point. Runs the full pipeline on a single image or folder.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Run the full three-stage pest detection pipeline")
    parser.add_argument("--binary_model", type=str,
                        default="models/checkpoints/binary_insect_classifier/weights/best.pt")
    parser.add_argument("--species_model", type=str,
                        default="models/checkpoints/best.pt")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single image or folder of images")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_trap_detection", action="store_true",
                        help="Skip Stage 0 and run directly on input image")
    args = parser.parse_args()

    pipeline = PestPipeline(
        args.binary_model, args.species_model,
        args.threshold, args.skip_trap_detection
    )

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob("*.jpg"))

    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    results = pipeline.classify_batch(image_paths)

    for r in results:
        name = Path(r["image"]).name
        if r["stage_reached"] == "no_trap_found":
            print(f"  {name:30s} -> NO TRAP DETECTED")
        elif not r["is_insect"]:
            print(f"  {name:30s} -> TRAP FOUND, no insect "
                  f"(conf {r['insect_confidence']:.2f})")
        else:
            print(f"  {name:30s} -> TRAP FOUND -> INSECT "
                  f"(conf {r['insect_confidence']:.2f}) "
                  f"-> {r['species']} (conf {r['species_confidence']:.2f})")

    n_trap = sum(1 for r in results if r["trap_detected"])
    n_insect = sum(1 for r in results if r["is_insect"])
    print(f"\nSummary:")
    print(f"  Traps detected : {n_trap}/{len(results)}")
    print(f"  Insects found  : {n_insect}/{n_trap} trap images")
    print(f"  Species IDed   : {n_insect} images passed to species model")


if __name__ == "__main__":
    main()