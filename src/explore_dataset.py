import os
from collections import Counter

# Paths
raw_dir = "data/raw/ip102_v1.1"
classes_file = "data/raw/Classification/classes.txt"

# Load class names
classes = {}
with open(classes_file) as f:
    for line in f:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            classes[int(parts[0]) - 1] = parts[1]

# Load splits
def load_split(filename):
    samples = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                samples.append((parts[0], int(parts[1])))
    return samples

train = load_split(os.path.join(raw_dir, "train.txt"))
val   = load_split(os.path.join(raw_dir, "val.txt"))
test  = load_split(os.path.join(raw_dir, "test.txt"))

print(f"Train images : {len(train)}")
print(f"Val images   : {len(val)}")
print(f"Test images  : {len(test)}")
print(f"Total        : {len(train) + len(val) + len(test)}")
print(f"Classes      : {len(classes)}")

# Class distribution in training set
counter = Counter(label for _, label in train)
print("\nTop 10 most common pests in training set:")
for cls_id, count in counter.most_common(10):
    print(f"  {count:>5} images  —  {classes.get(cls_id, 'unknown')}")

print("\nTop 10 least common pests in training set:")
for cls_id, count in counter.most_common()[:-11:-1]:
    print(f"  {count:>5} images  —  {classes.get(cls_id, 'unknown')}")