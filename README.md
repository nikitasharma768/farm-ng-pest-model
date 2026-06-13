# farm-ng-pest-model

A real-time pest detection and classification system deployed on the [farm-ng Amiga](https://farm-ng.com/) autonomous micro-tractor. The system automatically locates yellow sticky traps in the field, extracts them, classifies pests found on them, and generates a heatmap showing pest density and species distribution across the farm.

## Pipeline

```
Amiga camera footage
        ↓
Stage 1: Detect yellow trap (image processing)
        ↓
Stage 2: Extract trap region (crop)
        ↓
Stage 3: Classify pests on trap (YOLOv8)
        ↓
Output: Heatmap + species count per location
```

## Stages

**Stage 1 — Yellow trap detection**
Use HSV color filtering and contour detection (OpenCV) to locate yellow sticky traps in the Amiga camera feed. Each trap is cropped and passed to the classifier.

**Stage 2 — Pest classification**
A YOLOv8 model trained on the IP102 dataset detects and classifies pest species on the extracted trap image. The model covers 102 pest species across multiple crop types.

**Stage 3 — Field validation**
The full pipeline is validated by running the Amiga through an open field with real traps deployed. Performance is evaluated end-to-end.

**Output**
- Per-trap pest species count
- Heatmap showing trap locations and pest density across the field
- Combined species and location report

## Hardware Platform

The [farm-ng Amiga](https://farm-ng.com/) is a modular, all-electric micro-tractor with an open developer SDK and support for custom camera and compute configurations. The pest detection pipeline runs onboard as the Amiga traverses farm rows.

**Prior art:** CMU's TartanPest project (2023) used the same Amiga platform to detect spotted lanternfly egg masses using a deep learning model — a direct precedent for this work.

## Project Structure

```
farm-ng-pest-model/
├── data/
│   ├── raw/                  # IP102 dataset (not tracked by git)
│   ├── processed/            # Formatted data for training
│   └── validation/           # Trap images from test site
├── models/
│   ├── checkpoints/          # Saved model weights
│   └── exports/              # Exported models for Amiga deployment
├── notebooks/                # Exploration and analysis notebooks
├── src/
│   ├── trap_detector.py      # Stage 1: yellow trap detection
│   ├── pest_classifier.py    # Stage 2: YOLOv8 pest classification
│   ├── pipeline.py           # End-to-end inference pipeline
│   ├── heatmap.py            # Output: heatmap + species count
│   ├── explore_dataset.py    # Dataset exploration script
│   └── visualize_samples.py  # Sample image visualization
├── tests/
├── requirements.txt
└── README.md
```

## Dataset

| Dataset | Role | Classes |
|---------|------|---------|
| [IP102](https://github.com/xpwu95/IP102) | Primary training | 102 pest species, multi-crop |
| Trap images (test site) | Validation | Real field conditions |

**IP102 summary:**
- Total images: 75,222
- Train: 45,095 / Val: 7,508 / Test: 22,619
- Most represented: Cicadellidae (3,444 images)
- Least represented: Erythroneura apicalis (42 images)

## Model

- **Architecture:** YOLOv8 (classification mode)
- **Training data:** IP102 (102 pest species, multi-crop)
- **Input:** Cropped yellow trap image from Amiga camera
- **Output:** Detected species + confidence scores per trap

## Setup

```bash
git clone https://github.com/nikitasharma768/farm-ng-pest-model.git
cd farm-ng-pest-model
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Related Work

- [TartanPest (CMU, 2023)](https://www.cs.cmu.edu/news/2023/tartan-pest)
- [farm-ng Amiga Developer Docs](https://amiga.farm-ng.com)
- [IP102 Dataset](https://github.com/xpwu95/IP102)

## Acknowledgements

Developed under the supervision of Prof. Justin Morris.
Hardware platform: [farm-ng](https://farm-ng.com/).
