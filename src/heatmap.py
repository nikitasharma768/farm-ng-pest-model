"""
heatmap.py

Generates a heatmap and species count report from pest detection results.
Shows trap locations across the field with pest density visualized by
marker size and color.

Build order (per advisor guidance):
  MVP   : Static PNG heatmap with trap locations and pest counts
  Next  : Interactive HTML map
  Later : Clickable markers with full species breakdown per trap

Usage:
    python src/heatmap.py --input results/detection_log.json --output results/
    python src/heatmap.py --demo  (runs with simulated data to test output)
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def simulate_detection_log(n_traps=12):
    """
    Generate a simulated detection log for testing the heatmap output
    before real Amiga GPS data is available.

    Purpose:
        Allows the heatmap to be fully built and tested without needing
        field data. Real GPS coordinates and species detections from the
        Amiga pipeline will replace this once field validation begins.

    Args:
        n_traps (int): Number of trap locations to simulate.

    Returns:
        list: List of trap result dicts, each containing GPS coordinates,
              trap detection status, and species detections.
    """
    random.seed(42)

    # Simulated field: GPS coordinates around a central point
    base_lat = 34.0522
    base_lon = -117.2437

    species_pool = [
        "aphids", "corn_borer", "blister_beetle", "army_worm",
        "rice_leafhopper", "white_backed_plant_hopper", "Miridae",
        "cabbage_army_worm", "mole_cricket", "beet_army_worm"
    ]

    log = []
    for i in range(n_traps):
        lat = base_lat + random.uniform(-0.002, 0.002)
        lon = base_lon + random.uniform(-0.003, 0.003)
        n_insects = random.randint(0, 8)
        detections = []
        for _ in range(n_insects):
            detections.append({
                "species": random.choice(species_pool),
                "confidence": round(random.uniform(0.3, 0.95), 2)
            })

        log.append({
            "trap_id": f"trap_{i+1:02d}",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "trap_detected": True,
            "insect_count": n_insects,
            "detections": detections
        })

    return log


def load_detection_log(json_path):
    """
    Load a real detection log saved by pipeline.py.

    Args:
        json_path (str): Path to the detection log JSON file.

    Returns:
        list: List of trap result dicts.
    """
    with open(json_path) as f:
        data = json.load(f)
    return data.get("results", data)


def generate_png_heatmap(log, output_dir):
    """
    Generate a static PNG heatmap showing trap locations and pest density.

    Purpose:
        Creates a top-down field map (MVP output) where each trap is shown
        as a circle. Circle size and color represent the number of insects
        detected. A species count summary is printed alongside the map.
        This is the minimum viable output before adding HTML interactivity.

    Args:
        log        (list): List of trap result dicts with GPS coordinates
                           and detection counts.
        output_dir (Path): Folder to save the output PNG file.

    Returns:
        Path: Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lats = [t["latitude"] for t in log]
    lons = [t["longitude"] for t in log]
    counts = [t["insect_count"] for t in log]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#F0F7F4")

    # Left panel: heatmap
    ax = axes[0]
    ax.set_facecolor("#E8F5E9")
    ax.set_title("Field Pest Density Heatmap", fontsize=16, fontweight="bold", pad=15)

    max_count = max(counts) if max(counts) > 0 else 1
    colors = plt.cm.YlOrRd([c / max_count for c in counts])
    sizes = [max(80, c * 120) for c in counts]

    scatter = ax.scatter(lons, lats, s=sizes, c=counts,
                         cmap="YlOrRd", alpha=0.85,
                         edgecolors="gray", linewidths=0.8,
                         vmin=0, vmax=max_count)

    for t in log:
        ax.annotate(
            f"{t['trap_id']}\n({t['insect_count']} insects)",
            xy=(t["longitude"], t["latitude"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8, color="#1B3A2D",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor="none")
        )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("Insect count per trap", fontsize=11)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    zero_patch = mpatches.Patch(color=plt.cm.YlOrRd(0), label="No insects")
    low_patch = mpatches.Patch(color=plt.cm.YlOrRd(0.3), label="Low (1-3)")
    med_patch = mpatches.Patch(color=plt.cm.YlOrRd(0.6), label="Medium (4-6)")
    high_patch = mpatches.Patch(color=plt.cm.YlOrRd(1.0), label="High (7+)")
    ax.legend(handles=[zero_patch, low_patch, med_patch, high_patch],
              loc="lower right", fontsize=9, title="Pest density")

    # Right panel: species count bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#E8F5E9")
    ax2.set_title("Species Count Across All Traps", fontsize=16,
                  fontweight="bold", pad=15)

    all_species = []
    for t in log:
        for d in t.get("detections", []):
            all_species.append(d["species"])

    if all_species:
        species_counter = Counter(all_species)
        species_names = [s.replace("_", " ") for s in species_counter.keys()]
        species_counts = list(species_counter.values())

        sorted_pairs = sorted(zip(species_counts, species_names), reverse=True)
        species_counts, species_names = zip(*sorted_pairs)

        colors_bar = plt.cm.Set2(np.linspace(0, 1, len(species_names)))
        bars = ax2.barh(species_names, species_counts, color=colors_bar,
                        edgecolor="white", height=0.6)

        for bar, count in zip(bars, species_counts):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     str(count), va="center", fontsize=10, color="#1B3A2D")

        ax2.set_xlabel("Number of detections", fontsize=11)
        ax2.set_xlim(0, max(species_counts) * 1.2)
        ax2.grid(True, axis="x", alpha=0.3, linestyle="--")
    else:
        ax2.text(0.5, 0.5, "No insects detected", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14, color="gray")

    total_insects = sum(counts)
    total_traps = len(log)
    traps_with_insects = sum(1 for t in log if t["insect_count"] > 0)
    fig.suptitle(
        f"Pest Monitoring Report  |  {total_traps} traps scanned  |  "
        f"{total_insects} insects detected  |  "
        f"{traps_with_insects}/{total_traps} traps positive  |  "
        f"{datetime.now().strftime('%Y-%m-%d')}",
        fontsize=12, y=0.02, color="#1B3A2D"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = output_dir / "pest_heatmap.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"Heatmap saved to {out_path}")
    return out_path


def print_species_summary(log):
    """
    Print a text summary of species counts across all traps.

    Purpose:
        Quick readable output showing total insects, traps positive,
        and per-species breakdown, useful for logging and reporting
        without needing to open the PNG.

    Args:
        log (list): List of trap result dicts.

    Returns:
        None
    """
    total = sum(t["insect_count"] for t in log)
    positive = sum(1 for t in log if t["insect_count"] > 0)

    print(f"\nField Report Summary")
    print(f"{'='*40}")
    print(f"Total traps scanned : {len(log)}")
    print(f"Traps with insects  : {positive}/{len(log)}")
    print(f"Total insects found : {total}")

    all_species = []
    for t in log:
        for d in t.get("detections", []):
            all_species.append(d["species"])

    if all_species:
        print(f"\nSpecies breakdown:")
        for species, count in Counter(all_species).most_common():
            print(f"  {species.replace('_', ' '):35s}: {count}")
    else:
        print("\nNo insects detected across any trap.")


def main():
    """
    Main entry point. Loads detection data (real or simulated) and
    generates the PNG heatmap and species count output.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Generate pest detection heatmap")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to detection log JSON file from pipeline")
    parser.add_argument("--output", type=str, default="results",
                        help="Folder to save heatmap output")
    parser.add_argument("--demo", action="store_true",
                        help="Run with simulated data to test the output")
    args = parser.parse_args()

    if args.demo or args.input is None:
        print("Running with simulated GPS data (demo mode)...")
        log = simulate_detection_log(n_traps=12)
    else:
        print(f"Loading detection log from {args.input}...")
        log = load_detection_log(args.input)

    print_species_summary(log)
    generate_png_heatmap(log, args.output)


if __name__ == "__main__":
    main()