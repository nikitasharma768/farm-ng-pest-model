"""
make_pipeline_sample.py

Creates a small sample from the species test set for quickly testing
evaluate_pipeline.py before running it on the full 22,619-image set.
"""

from pathlib import Path
import shutil


def main():
    src = Path("data/processed/test")
    dst = Path("data/pipeline_eval_sample")
    dst.mkdir(exist_ok=True)

    species_folders = list(src.iterdir())[:5]

    for species_folder in species_folders:
        if species_folder.is_dir():
            out = dst / species_folder.name
            out.mkdir(exist_ok=True)
            images = list(species_folder.glob("*.jpg"))[:5]
            for img in images:
                shutil.copy(img, out / img.name)

    print(f"Sample created at {dst}")


if __name__ == "__main__":
    main()