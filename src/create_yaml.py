"""
create_yaml.py

Generates the data.yaml configuration file required by YOLOv8 classification
training. The file tells YOLOv8 where the processed dataset lives and what
the 102 pest class names are.

Output:
    data.yaml at the project root
"""

from pathlib import Path
import yaml


def load_classes(class_file):
    """
    Load and clean pest class names from classes.txt.

    Purpose:
        Reads the IP102 class index-to-name mapping and cleans names
        so they match the folder names created by prepare_dataset.py.

    Args:
        class_file (Path): Path to classes.txt. Lines formatted as
                           "1  rice leaf roller", 1-indexed.

    Returns:
        list: Ordered list of 102 class name strings (0-indexed),
              with spaces and slashes replaced by underscores.

    Example:
        ["rice_leaf_roller", "rice_leaf_caterpillar", ...]
    """
    classes = {}
    with open(class_file) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                idx  = int(parts[0]) - 1
                name = parts[1].strip()
                name = name.replace(" ", "_").replace("/", "_")
                classes[idx] = name
    return [classes[i] for i in sorted(classes)]


def create_yaml(output_path, processed_dir, class_names):
    """
    Write the data.yaml file for YOLOv8 classification training.

    Purpose:
        Creates the configuration file YOLOv8 reads at the start of
        training to find the dataset and class definitions.

    Args:
        output_path   (Path): Where to save the yaml file (project root).
        processed_dir (Path): Path to data/processed/ folder containing
                              train/, val/, and test/ subfolders.
        class_names   (list): Ordered list of class name strings.

    Returns:
        None

    Side effects:
        Writes data.yaml to output_path.
    """
    data = {
        "path"  : str(processed_dir.resolve()),
        "train" : "train",
        "val"   : "val",
        "test"  : "test",
        "nc"    : len(class_names),
        "names" : class_names
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"data.yaml created at {output_path}")
    print(f"  Classes : {len(class_names)}")
    print(f"  Train   : {processed_dir}/train")
    print(f"  Val     : {processed_dir}/val")
    print(f"  Test    : {processed_dir}/test")


def main():
    """
    Main entry point for yaml generation.

    Purpose:
        Loads class names and writes data.yaml to the project root.

    Args:
        None

    Returns:
        None
    """
    class_file    = Path("data/raw/Classification/classes.txt")
    processed_dir = Path("data/processed")
    output_path   = Path("data.yaml")

    class_names = load_classes(class_file)
    create_yaml(output_path, processed_dir, class_names)


if __name__ == "__main__":
    main()