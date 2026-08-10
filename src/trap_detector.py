"""
trap_detector.py

Stage 0 of the pest detection pipeline: locates yellow sticky traps in
images captured by the farm-ng Amiga camera using HSV color filtering
and contour detection. Crops the detected trap region and returns it
for downstream classification.

This stage runs before any ML model - it uses classical image processing
(OpenCV) to find the trap, so no training data is needed. It works in
both trap monitoring mode (finding deployed traps in the field) and can
be adapted for direct field scanning.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


# HSV color range for yellow sticky traps.
# Hue 20-35 covers yellow. Saturation >80 excludes washed-out/pale colors.
# Hue 100-130 covers blue, Saturation >80 excludes washed-out/pale colors.
# Value >80 excludes dark shadows.
# These values may need tuning based on lighting conditions in the field.
YELLOW_LOWER = np.array([15, 20, 80])
YELLOW_UPPER = np.array([40, 255, 255])

BLUE_LOWER = np.array([100, 20, 80])
BLUE_UPPER = np.array([130, 255, 255])

# Minimum and maximum contour area (in pixels) to be considered a trap.
# Filters out noise (too small) and irrelevant large yellow regions (too large).
MIN_TRAP_AREA = 5000
MAX_TRAP_AREA = 2000000


def detect_trap(image_path, debug=False):
    """
    Locate and crop the yellow sticky trap region from a single image.

    Purpose:
        Converts the image to HSV color space, applies a yellow color
        mask, finds contours in the mask, filters by size and shape,
        and returns the cropped trap region. This is the entry point
        for the full pipeline - the cropped region gets passed to the
        binary insect filter and then the species classifier.

    Args:
        image_path (str or Path): Path to the input image file.
        debug      (bool): If True, saves intermediate debug images
                           (the color mask and the annotated original)
                           to help diagnose detection issues in the field.

    Returns:
        tuple: (cropped_trap, bbox) where:
            cropped_trap (np.ndarray or None): The cropped trap image as
                a numpy array (BGR), or None if no trap was detected.
            bbox (tuple or None): (x, y, w, h) bounding box of the
                detected trap in the original image, or None if not found.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return None, None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    mask = cv2.bitwise_or(yellow_mask, blue_mask)

    # Clean up the mask with morphological operations to fill holes
    # and remove small noise specks before finding contours
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if debug:
            print("No contours found in mask")
        return None, None

    # Filter contours by area and pick the largest valid one
    valid_contours = [
        c for c in contours
        if MIN_TRAP_AREA < cv2.contourArea(c) < MAX_TRAP_AREA
    ]

    if not valid_contours:
        largest_area = max(cv2.contourArea(c) for c in contours)
        img_area = image.shape[0] * image.shape[1]

        # If yellow fills more than 50% of the frame, it's a close-up trap photo
        # Return the full image as the crop rather than failing
        if largest_area > img_area * 0.3:
            if debug:
                print(f"Trap fills full frame (area {largest_area:.0f}), returning full image")
            return image, (0, 0, image.shape[1], image.shape[0])

        if debug:
            print(f"No contours passed size filter (min={MIN_TRAP_AREA}, max={MAX_TRAP_AREA})")
            print(f"Largest contour area: {largest_area:.0f}")
        return None, None

    # Filter contours by area and pick the largest valid one
    valid_contours = [
        c for c in contours
        if MIN_TRAP_AREA < cv2.contourArea(c) < MAX_TRAP_AREA
    ]

    if not valid_contours:
        if debug:
            print(f"No contours passed size filter (min={MIN_TRAP_AREA}, max={MAX_TRAP_AREA})")
            print(f"Largest contour area: {max(cv2.contourArea(c) for c in contours):.0f}")
        return None, None

    largest = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cropped = image[y:y+h, x:x+w]

    if debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(debug_img, f"Trap detected ({w}x{h}px)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        debug_path = Path(image_path).parent / f"debug_{Path(image_path).name}"
        cv2.imwrite(str(debug_path), debug_img)
        mask_path = Path(image_path).parent / f"mask_{Path(image_path).name}"
        cv2.imwrite(str(mask_path), mask)
        print(f"Debug images saved: {debug_path}, {mask_path}")

    return cropped, (x, y, w, h)


def process_folder(input_dir, output_dir, debug=False):
    """
    Run trap detection on every image in a folder and save cropped results.

    Purpose:
        Batch processing entry point for testing the trap detector on a
        folder of images before connecting it to the full pipeline.
        Saves successfully cropped trap images to the output directory.

    Args:
        input_dir  (Path): Folder containing input images.
        output_dir (Path): Folder to save cropped trap images.
        debug      (bool): If True, saves debug visualizations alongside
                           each processed image.

    Returns:
        tuple: (detected, total) counts of successful detections and
               total images processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    detected = 0
    for img_path in images:
        cropped, bbox = detect_trap(img_path, debug=debug)
        if cropped is not None:
            out_path = output_dir / f"trap_{img_path.name}"
            cv2.imwrite(str(out_path), cropped)
            detected += 1
            print(f"  {img_path.name}: trap detected at {bbox}")
        else:
            print(f"  {img_path.name}: no trap found")

    return detected, len(images)


def main():
    """
    Main entry point. Runs trap detection on a single image or folder.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Detect and crop yellow sticky traps")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single image or folder of images")
    parser.add_argument("--output", type=str, default="data/trap_crops",
                        help="Folder to save cropped trap images")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug visualizations showing detected regions")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        cropped, bbox = detect_trap(input_path, debug=args.debug)
        if cropped is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"trap_{input_path.name}"
            cv2.imwrite(str(out_file), cropped)
            print(f"Trap detected at {bbox}, saved to {out_file}")
        else:
            print("No trap detected in this image.")
    else:
        detected, total = process_folder(input_path, output_path, debug=args.debug)
        print(f"\nDone: {detected}/{total} images had a trap detected")
        print(f"Cropped traps saved to {output_path}")


if __name__ == "__main__":
    main()