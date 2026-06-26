"""
check_imagenet_classes.py

Maps the WordNet IDs (WNIDs) used by timm/mini-imagenet to human-readable
class names, using the standard ImageNet WNID-to-name mapping. This lets us
manually identify and exclude any insect-related classes before using this
dataset as the negative ("not insect") class for the binary classifier.
"""

import json
import urllib.request
from datasets import load_dataset


def get_wnid_to_name_map():
    """
    Download the standard ImageNet class index and build a WNID to name map.

    Purpose:
        The timm/mini-imagenet dataset only provides WordNet IDs (e.g.
        "n01532829"), not readable names. This downloads the well-known
        Keras/TensorFlow ImageNet class index file and rebuilds it as a
        dictionary keyed by WNID instead of its original integer-string key,
        so we can look up any WNID directly.

    Args:
        None

    Returns:
        dict: Mapping of WNID string (e.g. "n01532829") to its full
              English class name (e.g. "brambling").
    """
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as response:
        raw = json.loads(response.read())

    wnid_map = {}
    for key in raw:
        entry = raw[key]
        wnid = entry[0]
        name = entry[1]
        wnid_map[wnid] = name
    return wnid_map


def get_mini_imagenet_wnids():
    """
    Load the timm/mini-imagenet dataset and extract its WNID class labels.

    Purpose:
        Retrieves the 100 WordNet IDs used as class labels in this dataset,
        in label-index order, so we can map them to readable names.

    Args:
        None

    Returns:
        list: List of WNID strings, in label-index order.
    """
    dataset = load_dataset("timm/mini-imagenet", split="train")
    label_feature = dataset.features["label"]
    return label_feature.names


def main():
    """
    Main entry point. Maps each mini-imagenet WNID to its readable name and
    prints the full list for manual review, flagging likely insect classes.

    Args:
        None

    Returns:
        None
    """
    print("Downloading WNID-to-name mapping...")
    wnid_map = get_wnid_to_name_map()
    print("Loaded " + str(len(wnid_map)) + " WNID mappings")

    print("Loading mini-imagenet class list...")
    wnids = get_mini_imagenet_wnids()

    insect_keywords = [
        "bee", "ant", "beetle", "fly", "moth", "butterfly", "grasshopper",
        "cricket", "dragonfly", "damselfly", "cicada", "aphid", "wasp",
        "cockroach", "mantis", "weevil", "ladybug", "ladybird", "termite",
        "lacewing", "leafhopper", "katydid"
    ]

    print("")
    print("Total classes: " + str(len(wnids)))
    print("")

    flagged = []
    not_found = []

    for i in range(len(wnids)):
        wnid = wnids[i]
        name = wnid_map.get(wnid)
        if name is None:
            not_found.append(wnid)
            name = "NOT FOUND"

        name_lower = name.lower()
        is_insect = False
        for kw in insect_keywords:
            if kw in name_lower:
                is_insect = True

        flag = ""
        if is_insect:
            flag = "  <-- POSSIBLE INSECT"
            flagged.append((i, wnid, name))

        print("  " + str(i) + ": " + wnid + "  " + name + flag)

    print("")
    print(str(len(flagged)) + " classes flagged as possible insects:")
    for item in flagged:
        print("  " + str(item[0]) + ": " + item[1] + "  " + item[2])

    if len(not_found) > 0:
        print("")
        print(str(len(not_found)) + " WNIDs not found in standard mapping:")
        print(not_found)


if __name__ == "__main__":
    main()