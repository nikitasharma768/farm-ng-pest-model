import cv2
import numpy as np
from pathlib import Path
import argparse

def sample_hsv(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read: {image_path}")
        return
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    print(f"\nImage: {Path(image_path).name} ({w}x{h})")
    regions = [
        ("Top-left",     hsv[h//8:h//4,     w//8:w//4]),
        ("Top-right",    hsv[h//8:h//4,     w*3//4:w*7//8]),
        ("Center",       hsv[h//3:h*2//3,   w//3:w*2//3]),
        ("Bottom-left",  hsv[h*3//4:h*7//8, w//8:w//4]),
        ("Bottom-right", hsv[h*3//4:h*7//8, w*3//4:w*7//8]),
    ]
    for name, region in regions:
        print(f"  {name:15s}: H={np.mean(region[:,:,0]):.1f}  S={np.mean(region[:,:,1]):.1f}  V={np.mean(region[:,:,2]):.1f}")
    print(f"  Full image H range: {hsv[:,:,0].min()} to {hsv[:,:,0].max()}")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()

input_path = Path(args.input)
if input_path.is_file():
    sample_hsv(input_path)
else:
    for img in sorted(input_path.glob("*.jpg"))[:5]:
        sample_hsv(img)