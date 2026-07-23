"""
run_all.py

Single command to run the complete pest detection system end to end:
  1. Runs the three-stage pipeline on a folder of images
  2. Saves a JSON detection log
  3. Generates the PNG heatmap
  4. Generates the interactive HTML heatmap

Usage:
    python3 src/run_all.py --input data/validation/trap_photos
    python3 src/run_all.py --input your/image/folder --output results/
    python3 src/run_all.py --input your/folder --skip_trap_detection
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """
    Main entry point. Runs pipeline then both heatmap scripts in sequence.

    Purpose:
        Provides a single command that executes the full system from
        raw images to heatmap output, so the complete pipeline can be
        run with one command rather than three separate scripts.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Run the full pest detection system end to end")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to folder of images to process")
    parser.add_argument("--output", type=str, default="results",
                        help="Folder to save all outputs")
    parser.add_argument("--binary_model", type=str,
                        default="models/checkpoints/binary_insect_classifier/weights/best.pt")
    parser.add_argument("--species_model", type=str,
                        default="models/checkpoints/best.pt")
    parser.add_argument("--skip_trap_detection", action="store_true",
                        help="Skip trap detector, run directly on images")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "detection_log.json"

    print("=" * 55)
    print("  Autonomous Pest Detection System")
    print("  farm-ng Amiga Pipeline")
    print("=" * 55)

    print("\nStep 1/3: Running detection pipeline...")
    pipeline_cmd = [
        sys.executable, "src/pipeline.py",
        "--input", args.input,
        "--binary_model", args.binary_model,
        "--species_model", args.species_model,
    ]
    if args.skip_trap_detection:
        pipeline_cmd.append("--skip_trap_detection")

    result = subprocess.run(pipeline_cmd)
    if result.returncode != 0:
        print("Pipeline failed. Stopping.")
        return

    print("\nStep 2/3: Generating PNG heatmap...")
    subprocess.run([
        sys.executable, "src/heatmap.py",
        "--input", str(log_path),
        "--output", str(output_dir)
    ])

    print("\nStep 3/3: Generating interactive HTML heatmap...")
    subprocess.run([
        sys.executable, "src/heatmap_html.py",
        "--input", str(log_path),
        "--output", str(output_dir)
    ])

    print("\n" + "=" * 55)
    print("  Done!")
    print(f"  Detection log : {log_path}")
    print(f"  PNG heatmap   : {output_dir}/pest_heatmap.png")
    print(f"  HTML heatmap  : {output_dir}/pest_heatmap.html")
    print("=" * 55)


if __name__ == "__main__":
    main()
