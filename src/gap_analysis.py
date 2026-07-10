"""
gap_analysis.py

Analyzes the IP102 dataset to identify gaps that limit the species
classifier's real-world performance. Documents which classes have too
few training images, which crop types are underrepresented, and what
a more complete dataset would need to include.

This is a research contribution: it defines what a deployment-ready
cross-crop pest dataset should look like, and quantifies the gap
between IP102 and that ideal.

Output:
    results/gap_analysis.json  - full structured findings
    results/gap_analysis.png   - visual summary charts
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict


# IP102 crop type groupings based on the dataset's own taxonomy.
# Each pest class belongs to a primary crop category.
CROP_GROUPS = {
    "Rice": [
        "rice_leaf_roller", "rice_leaf_caterpillar", "paddy_stem_maggot",
        "asiatic_rice_borer", "yellow_rice_borer", "rice_gall_midge",
        "Rice_Stemfly", "brown_plant_hopper", "white_backed_plant_hopper",
        "small_brown_plant_hopper", "rice_water_weevil", "rice_leafhopper",
        "grain_spreader_thrips", "rice_shell_pest"
    ],
    "Corn/Wheat": [
        "grub", "mole_cricket", "wireworm", "white_margined_moth",
        "black_cutworm", "large_cutworm", "yellow_cutworm", "red_spider",
        "corn_borer", "army_worm", "aphids", "Potosiabre_vitarsis",
        "peach_borer", "english_grain_aphid", "green_bug",
        "bird_cherry-oataphid", "wheat_blossom_midge", "penthaleus_major",
        "longlegged_spider_mite", "wheat_phloeothrips", "wheat_sawfly",
        "cerodonta_denticornis"
    ],
    "Beet/Vegetable": [
        "beet_fly", "flea_beetle", "cabbage_army_worm", "beet_army_worm",
        "Beet_spot_flies", "meadow_moth", "beet_weevil",
        "sericaorient_alismots_chulsky"
    ],
    "Alfalfa/Legume": [
        "alfalfa_weevil", "flax_budworm", "alfalfa_plant_bug",
        "tarnished_plant_bug", "Locustoidea", "lytta_polita",
        "legume_blister_beetle", "blister_beetle",
        "therioaphis_maculata_Buckton", "odontothrips_loti", "Thrips",
        "alfalfa_seed_chalcid"
    ],
    "Vitis/Grape": [
        "Pieris_canidia", "Apolygus_lucorum", "Limacodidae",
        "Viteus_vitifoliae", "Colomerus_vitis",
        "Brevipoalpus_lewisi_McGregor", "oides_decempunctata",
        "Polyphagotars_onemus_latus", "Pseudococcus_comstocki_Kuwana",
        "parathrene_regalis", "Ampelophaga"
    ],
    "Citrus": [
        "Lycorma_delicatula", "Xylotrechus", "Cicadella_viridis",
        "Miridae", "Trialeurodes_vaporariorum", "Erythroneura_apicalis",
        "Papilio_xuthus", "Panonchus_citri_McGregor",
        "Phyllocoptes_oleiverus_ashmead", "Icerya_purchasi_Maskell",
        "Unaspis_yanonensis", "Ceroplastes_rubens",
        "Chrysomphalus_aonidum", "Parlatoria_zizyphus_Lucus",
        "Nipaecoccus_vastalor", "Aleurocanthus_spiniferus",
        "Tetradacus_c_Bactrocera_minax", "Dacus_dorsalis(Hendel)",
        "Bactrocera_tsuneonis", "Prodenia_litura", "Adristyrannus",
        "Phyllocnistis_citrella_Stainton", "Toxoptera_citricidus",
        "Toxoptera_aurantii", "Aphis_citricola_Vander_Goot",
        "Scirtothrips_dorsalis_Hood", "Dasineura_sp",
        "Lawana_imitata_Melichar", "Salurnis_marginella_Guerr"
    ],
    "Mango": [
        "Deporaus_marginatus_Pascoe", "Chlumetia_transversa",
        "Mango_flat_beak_leafhopper", "Rhytidodera_bowrinii_white",
        "Sternochetus_frigidus"
    ],
    "General/Other": [
        "Cicadellidae"
    ]
}

# Minimum images per class considered adequate for reliable classification
ADEQUATE_THRESHOLD = 200
# Below this is critically underrepresented
CRITICAL_THRESHOLD = 100


def load_class_counts(processed_dir):
    """
    Count training images per class from the processed dataset folder.

    Purpose:
        Reads the train split of the formatted dataset and counts how
        many images exist per species class. This is the ground truth
        for the gap analysis - we use the actual file counts rather than
        the txt file, since the processed folder reflects what the model
        actually trained on.

    Args:
        processed_dir (Path): Path to data/processed/ containing train/
                              val/ test/ subfolders.

    Returns:
        dict: Mapping of class name (str) to image count (int).
    """
    train_dir = processed_dir / "train"
    counts = {}
    for class_folder in sorted(train_dir.iterdir()):
        if class_folder.is_dir():
            counts[class_folder.name] = len(list(class_folder.glob("*.jpg")))
    return counts


def analyze_class_distribution(counts):
    """
    Identify which classes are critically or marginally underrepresented.

    Purpose:
        Classifies each of the 102 species into adequate, marginal, or
        critical based on training image count. This directly informs
        which species the model is likely to perform worst on, and which
        classes a better dataset should prioritize collecting more images
        for.

    Args:
        counts (dict): Mapping of class name to training image count.

    Returns:
        dict: {
            "adequate"  : list of (class, count) with >= ADEQUATE_THRESHOLD,
            "marginal"  : list of (class, count) between thresholds,
            "critical"  : list of (class, count) below CRITICAL_THRESHOLD,
            "stats"     : summary statistics dict
        }
    """
    adequate = []
    marginal = []
    critical = []

    for cls, count in sorted(counts.items(), key=lambda x: x[1]):
        if count >= ADEQUATE_THRESHOLD:
            adequate.append((cls, count))
        elif count >= CRITICAL_THRESHOLD:
            marginal.append((cls, count))
        else:
            critical.append((cls, count))

    all_counts = list(counts.values())
    return {
        "adequate": adequate,
        "marginal": marginal,
        "critical": critical,
        "stats": {
            "total_classes": len(counts),
            "total_images": sum(all_counts),
            "mean_per_class": round(sum(all_counts) / len(all_counts), 1),
            "median_per_class": int(sorted(all_counts)[len(all_counts) // 2]),
            "min_count": min(all_counts),
            "max_count": max(all_counts),
            "adequate_count": len(adequate),
            "marginal_count": len(marginal),
            "critical_count": len(critical),
        }
    }


def analyze_crop_coverage(counts):
    """
    Summarize class counts and image totals grouped by crop type.

    Purpose:
        Reveals which agricultural contexts are well represented vs.
        underrepresented in IP102. A model trained on an imbalanced
        crop distribution will generalize better to some farm types
        than others, which is a key limitation to document for the
        research writeup.

    Args:
        counts (dict): Mapping of class name to training image count.

    Returns:
        dict: Mapping of crop group name to {
            "classes": int,
            "total_images": int,
            "mean_per_class": float,
            "classes_below_threshold": int
        }
    """
    crop_stats = {}
    for crop, classes in CROP_GROUPS.items():
        class_counts = []
        below = 0
        for cls in classes:
            c = counts.get(cls, 0)
            class_counts.append(c)
            if c < ADEQUATE_THRESHOLD:
                below += 1
        crop_stats[crop] = {
            "classes": len(classes),
            "total_images": sum(class_counts),
            "mean_per_class": round(sum(class_counts) / len(class_counts), 1)
                              if class_counts else 0,
            "classes_below_threshold": below
        }
    return crop_stats


def generate_gap_report(dist, crop_stats, output_dir):
    """
    Save the full gap analysis findings as a structured JSON file.

    Purpose:
        Creates a machine-readable record of the gap analysis that can
        be referenced in the research writeup and shared with Prof. Morris.
        Includes per-class findings, crop-level summary, and specific
        recommendations for improving the dataset.

    Args:
        dist       (dict): Output from analyze_class_distribution().
        crop_stats (dict): Output from analyze_crop_coverage().
        output_dir (Path): Folder to save the JSON report.

    Returns:
        Path: Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": dist["stats"],
        "thresholds": {
            "adequate": ADEQUATE_THRESHOLD,
            "critical": CRITICAL_THRESHOLD
        },
        "class_distribution": {
            "critical_classes": [
                {"class": c, "count": n} for c, n in dist["critical"]
            ],
            "marginal_classes": [
                {"class": c, "count": n} for c, n in dist["marginal"]
            ],
        },
        "crop_coverage": crop_stats,
        "recommendations": [
            "Collect at least 200 additional images for each of the "
            f"{dist['stats']['critical_count']} critically underrepresented classes",
            "Mango pest classes have the fewest images per class on average "
            "— prioritize this crop type for new data collection",
            "Add generalist species (house flies, field crickets, beetles) "
            "not currently in IP102 to support real-world uncultivated land deployment",
            "Include images from Californian farms specifically to improve "
            "regional relevance for Amiga deployment in this geography",
            "Collect images from sticky trap surfaces specifically, rather than "
            "general field photos, to match the Amiga's actual capture conditions"
        ]
    }
    out_path = output_dir / "gap_analysis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Gap analysis report saved to {out_path}")
    return out_path


def generate_gap_charts(dist, crop_stats, output_dir):
    """
    Generate visual charts summarizing the gap analysis findings.

    Purpose:
        Creates a two-panel figure showing (1) the distribution of
        training images per class with threshold lines marked, and
        (2) a crop-type comparison bar chart. These visuals are suitable
        for the research poster and writeup.

    Args:
        dist       (dict): Output from analyze_class_distribution().
        crop_stats (dict): Output from analyze_crop_coverage().
        output_dir (Path): Folder to save the PNG chart.

    Returns:
        Path: Path to the saved PNG file.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#F0F7F4")

    # Left panel: class distribution histogram
    ax = axes[0]
    ax.set_facecolor("#E8F5E9")
    all_counts = (
        [c for _, c in dist["critical"]] +
        [c for _, c in dist["marginal"]] +
        [c for _, c in dist["adequate"]]
    )
    colors = (
        ["#C1121F"] * len(dist["critical"]) +
        ["#F4A261"] * len(dist["marginal"]) +
        ["#2D6A4F"] * len(dist["adequate"])
    )
    ax.bar(range(len(all_counts)), all_counts, color=colors, width=0.8)
    ax.axhline(y=ADEQUATE_THRESHOLD, color="#2D6A4F", linestyle="--",
               linewidth=1.5, label=f"Adequate threshold ({ADEQUATE_THRESHOLD})")
    ax.axhline(y=CRITICAL_THRESHOLD, color="#C1121F", linestyle="--",
               linewidth=1.5, label=f"Critical threshold ({CRITICAL_THRESHOLD})")
    ax.set_xlabel("Species classes (sorted by count)", fontsize=12)
    ax.set_ylabel("Training images", fontsize=12)
    ax.set_title("Training Images Per Species Class", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)

    red_p = mpatches.Patch(color="#C1121F",
                           label=f"Critical (<{CRITICAL_THRESHOLD}): "
                                 f"{len(dist['critical'])} classes")
    org_p = mpatches.Patch(color="#F4A261",
                           label=f"Marginal ({CRITICAL_THRESHOLD}-"
                                 f"{ADEQUATE_THRESHOLD}): "
                                 f"{len(dist['marginal'])} classes")
    grn_p = mpatches.Patch(color="#2D6A4F",
                           label=f"Adequate (>={ADEQUATE_THRESHOLD}): "
                                 f"{len(dist['adequate'])} classes")
    ax.legend(handles=[red_p, org_p, grn_p], fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Right panel: crop coverage
    ax2 = axes[1]
    ax2.set_facecolor("#E8F5E9")
    crops = list(crop_stats.keys())
    means = [crop_stats[c]["mean_per_class"] for c in crops]
    bar_colors = ["#C1121F" if m < CRITICAL_THRESHOLD
                  else "#F4A261" if m < ADEQUATE_THRESHOLD
                  else "#2D6A4F" for m in means]

    bars = ax2.barh(crops, means, color=bar_colors, edgecolor="white", height=0.6)
    ax2.axvline(x=ADEQUATE_THRESHOLD, color="#2D6A4F", linestyle="--",
                linewidth=1.5)
    ax2.axvline(x=CRITICAL_THRESHOLD, color="#C1121F", linestyle="--",
                linewidth=1.5)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"{m:.0f}", va="center", fontsize=10, color="#1B3A2D")
    ax2.set_xlabel("Mean training images per class", fontsize=12)
    ax2.set_title("Dataset Coverage by Crop Type", fontsize=14,
                  fontweight="bold")
    ax2.set_xlim(0, max(means) * 1.2)
    ax2.grid(True, axis="x", alpha=0.3, linestyle="--")

    fig.suptitle(
        f"IP102 Gap Analysis  |  {dist['stats']['total_classes']} classes  |  "
        f"{dist['stats']['total_images']:,} training images  |  "
        f"{len(dist['critical'])} critically underrepresented classes",
        fontsize=12, y=0.02, color="#1B3A2D"
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = output_dir / "gap_analysis.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Gap analysis chart saved to {out_path}")
    return out_path


def main():
    """
    Main entry point. Loads the processed dataset, runs gap analysis,
    and saves both JSON report and PNG chart.

    Args:
        None

    Returns:
        None
    """
    processed_dir = Path("data/processed")
    output_dir = Path("results")

    print("Loading class counts from processed dataset...")
    counts = load_class_counts(processed_dir)
    print(f"Found {len(counts)} classes")

    print("\nAnalyzing class distribution...")
    dist = analyze_class_distribution(counts)
    s = dist["stats"]
    print(f"  Total images   : {s['total_images']:,}")
    print(f"  Mean per class : {s['mean_per_class']}")
    print(f"  Min / Max      : {s['min_count']} / {s['max_count']}")
    print(f"  Adequate       : {s['adequate_count']} classes")
    print(f"  Marginal       : {s['marginal_count']} classes")
    print(f"  Critical       : {s['critical_count']} classes")

    print(f"\nCritically underrepresented classes "
          f"(fewer than {CRITICAL_THRESHOLD} training images):")
    for cls, count in dist["critical"]:
        print(f"  {cls:40s}: {count}")

    print("\nAnalyzing crop type coverage...")
    crop_stats = analyze_crop_coverage(counts)
    for crop, stats in sorted(crop_stats.items(),
                               key=lambda x: x[1]["mean_per_class"]):
        print(f"  {crop:20s}: {stats['mean_per_class']:6.1f} mean images/class "
              f"({stats['classes']} classes, "
              f"{stats['classes_below_threshold']} below threshold)")

    generate_gap_report(dist, crop_stats, output_dir)
    generate_gap_charts(dist, crop_stats, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()