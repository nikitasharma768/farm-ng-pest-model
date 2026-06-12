import os
import random
import matplotlib.pyplot as plt
from PIL import Image

raw_dir = "data/raw/ip102_v1.1"
img_dir = os.path.join(raw_dir, "images")
classes_file = "data/raw/Classification/classes.txt"

# Load class names
classes = {}
with open(classes_file) as f:
    for line in f:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            classes[int(parts[0]) - 1] = parts[1]

# Load train samples
samples = []
with open(os.path.join(raw_dir, "train.txt")) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            samples.append((parts[0], int(parts[1])))

# Pick 12 random samples
random.shuffle(samples)
selected = samples[:12]

# Plot
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for ax, (fname, label) in zip(axes.flatten(), selected):
    img_path = os.path.join(img_dir, fname)
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(classes.get(label, "unknown"), fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/processed/sample_images.png")
plt.show()
print("Saved to data/processed/sample_images.png")