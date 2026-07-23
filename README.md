# farm-ng-pest-model

An autonomous pest detection and classification system deployed on the
[farm-ng Amiga](https://farm-ng.com/) micro-tractor for continuous,
real-time pest monitoring in agricultural fields.

## Pipeline

```
Amiga Camera
      ↓
Stage 0: Trap Detector (HSV color filtering + contour detection)
      ↓
Stage A: Binary Insect Filter (YOLOv8n, F1: 97.0%)
      ↓
Stage B: Species Classifier (YOLOv8s, 102 classes, 71.6% top-1)
      ↓
Output: Heatmap + Species Count
```

The system operates in two modes:
- **Trap monitoring mode**: detects yellow sticky traps, crops them, classifies insects on the trap
- **Direct scan mode**: scans plants and rows directly without traps (`--skip_trap_detection`)

## Quick Start

```bash
git clone https://github.com/nikitasharma768/farm-ng-pest-model.git
cd farm-ng-pest-model
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Run the full pipeline on a folder of images:

```bash
python3 src/run_all.py --input your/image/folder
```

This runs all three stages and generates both PNG and HTML heatmap outputs.

## Results

| Model | Metric | Score |
|---|---|---|
| Binary insect filter | Precision | 95.0% |
| Binary insect filter | Recall | 99.1% |
| Binary insect filter | F1 Score | 97.0% |
| Species classifier | Top-1 accuracy | 71.6% |
| Species classifier | Top-5 accuracy | 91.4% |
| Combined pipeline | End-to-end top-1 | 71.5% |

**Taxonomic analysis:** 87.6% of misclassifications occur between species
in the same taxonomic family, and 100% within the same order. Model errors
are biologically clustered, not random.

## Project Structure

```
farm-ng-pest-model/
├── src/
│   ├── run_all.py              # Single command: run full pipeline + heatmaps
│   ├── pipeline.py             # End-to-end three-stage pipeline
│   ├── trap_detector.py        # Stage 0: yellow trap detection (OpenCV)
│   ├── train.py                # Train the 102-species classifier
│   ├── train_binary.py         # Train the binary insect classifier
│   ├── evaluate_binary.py      # Evaluate binary classifier with F1 score
│   ├── evaluate_pipeline.py    # Evaluate full combined pipeline
│   ├── heatmap.py              # PNG field heatmap output
│   ├── heatmap_html.py         # Interactive HTML heatmap output
│   ├── gap_analysis.py         # IP102 dataset gap analysis
│   ├── analyze_confusion.py    # Taxonomic confusion analysis
│   ├── taxonomy_check.py       # Verify confusion pairs against taxonomy
│   ├── prepare_dataset.py      # Format IP102 for YOLOv8 classification
│   ├── build_binary_dataset.py # Build binary insect/not-insect dataset
│   └── explore_dataset.py      # Dataset exploration and visualization
├── ros2_nodes/                 # ROS 2 package for Amiga deployment
│   ├── pest_detection_nodes/
│   │   ├── trap_detector_node.py      # ROS node: Stage 0
│   │   ├── binary_filter_node.py      # ROS node: Stage A
│   │   ├── species_classifier_node.py # ROS node: Stage B
│   │   └── heatmap_node.py            # ROS node: output
│   ├── launch/
│   │   └── pest_detection.launch.py   # Launch all nodes at once
│   ├── package.xml
│   └── setup.py
├── data/                       # Datasets (not tracked by git)
├── models/                     # Model weights (not tracked by git)
├── results/                    # Pipeline outputs
└── requirements.txt
```

## ROS 2 Deployment (farm-ng Amiga)

The pipeline is implemented as a ROS 2 Jazzy package for native Amiga deployment.

**Install ROS 2 package:**

```bash
mkdir -p ~/ros2_ws/src
cp -r ros2_nodes ~/ros2_ws/src/pest_detection_nodes
cd ~/ros2_ws
colcon build --packages-select pest_detection_nodes
source install/setup.bash
```

**Launch the full pipeline:**

```bash
ros2 launch pest_detection_nodes pest_detection.launch.py
```

**Node topics:**

| Node | Subscribes | Publishes |
|---|---|---|
| trap_detector_node | /camera/image | /trap/cropped_image |
| binary_filter_node | /trap/cropped_image | /insect/detected, /insect/image |
| species_classifier_node | /insect/image | /insect/species |
| heatmap_node | /insect/species, /gps/fix | /heatmap/status |

## Dataset

| Dataset | Role | Size |
|---|---|---|
| IP102 | Species classifier training | 75,222 images, 102 classes |
| mini-ImageNet (filtered) | Binary classifier negative class | ~48,500 images, 97 classes |
| Trap photos (test site) | Validation | 15 images |

**IP102 gap analysis:** 10 of 102 classes have fewer than 100 training
images. Mango and Vitis/Grape crop types are the most underrepresented.

## Related Work

- [TartanPest (CMU, 2023)](https://www.cs.cmu.edu/news/2023/tartan-pest)
- [farm-ng Amiga Developer Docs](https://amiga.farm-ng.com)
- [IP102 Dataset](https://github.com/xpwu95/IP102)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

## Acknowledgements

Developed under the supervision of Prof. Justin Morris, CSUSM.
Hardware platform: [farm-ng](https://farm-ng.com/).
