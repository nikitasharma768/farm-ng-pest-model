"""
taxonomy_check.py

Checks whether the most-confused species pairs (from analyze_confusion.py)
share the same taxonomic family or order. This turns a visual impression
("these look related") into a verifiable claim for the research writeup,
as requested by Prof. Morris.

Note: family/order assignments below are based on standard entomological
classification (verified manually), not the IP102 dataset's own grouping,
which groups by crop type rather than biological taxonomy.
"""

# Family/order lookup for species that appeared in the confusion analysis.
# Only species relevant to this analysis are included so each entry could
# be checked, rather than guessing at the full 102-class taxonomy.
TAXONOMY = {
    "blister_beetle":          ("Meloidae",     "Coleoptera"),
    "legume_blister_beetle":   ("Meloidae",     "Coleoptera"),
    "lytta_polita":            ("Meloidae",     "Coleoptera"),
    "english_grain_aphid":     ("Aphididae",    "Hemiptera"),
    "bird_cherry-oataphid":    ("Aphididae",    "Hemiptera"),
    "aphids":                  ("Aphididae",    "Hemiptera"),
    "green_bug":               ("Aphididae",    "Hemiptera"),
    "tarnished_plant_bug":     ("Miridae",       "Hemiptera"),
    "Miridae":                 ("Miridae",       "Hemiptera"),
    "alfalfa_plant_bug":       ("Miridae",       "Hemiptera"),
    "Apolygus_lucorum":        ("Miridae",       "Hemiptera"),
    "large_cutworm":           ("Noctuidae",     "Lepidoptera"),
    "black_cutworm":           ("Noctuidae",     "Lepidoptera"),
    "yellow_cutworm":          ("Noctuidae",     "Lepidoptera"),
    "cabbage_army_worm":       ("Noctuidae",     "Lepidoptera"),
    "beet_army_worm":          ("Noctuidae",     "Lepidoptera"),
    "army_worm":               ("Noctuidae",     "Lepidoptera"),
    "brown_plant_hopper":      ("Delphacidae",   "Hemiptera"),
    "white_backed_plant_hopper": ("Delphacidae", "Hemiptera"),
    "small_brown_plant_hopper": ("Delphacidae",  "Hemiptera"),
    "Cicadellidae":            ("Cicadellidae",  "Hemiptera"),
    "Cicadella_viridis":       ("Cicadellidae",  "Hemiptera"),
    "rice_leafhopper":         ("Cicadellidae",  "Hemiptera"),
    "rice_shell_pest":         ("Pyralidae",     "Lepidoptera"),
    "rice_leaf_roller":        ("Crambidae",     "Lepidoptera"),
}


def check_pair(true_label, pred_label):
    """
    Check whether two species share the same taxonomic family or order.

    Purpose:
        Looks up both species in the TAXONOMY table and reports whether
        they match at the family level, order level, or neither. This is
        the core check supporting the claim that misclassifications occur
        between biologically similar species.

    Args:
        true_label (str): The correct species name (matches IP102 class
                          folder naming, e.g. "blister_beetle").
        pred_label (str): The species the model predicted instead.

    Returns:
        dict: {
            "true": str, "pred": str,
            "true_family": str or None, "pred_family": str or None,
            "same_family": bool, "same_order": bool,
            "known": bool  # False if either species is missing from TAXONOMY
        }
    """
    true_info = TAXONOMY.get(true_label)
    pred_info = TAXONOMY.get(pred_label)

    if true_info is None or pred_info is None:
        return {
            "true": true_label, "pred": pred_label,
            "true_family": None, "pred_family": None,
            "same_family": False, "same_order": False,
            "known": False
        }

    true_family, true_order = true_info
    pred_family, pred_order = pred_info

    return {
        "true": true_label, "pred": pred_label,
        "true_family": true_family, "pred_family": pred_family,
        "same_family": true_family == pred_family,
        "same_order": true_order == pred_order,
        "known": True
    }


def main():
    """
    Main entry point. Checks the confusion pairs found in the earlier
    analysis against the taxonomy table and prints a summary.

    Purpose:
        Provides a verifiable count of how many of the top confused pairs
        are taxonomically related, to support the research claim that
        model errors cluster among similar species rather than being random.

    Args:
        None

    Returns:
        None
    """
    # Top confused pairs from analyze_confusion.py output (true, predicted, count)
    confused_pairs = [
        ("blister_beetle", "legume_blister_beetle", 71),
        ("english_grain_aphid", "bird_cherry-oataphid", 60),
        ("tarnished_plant_bug", "Miridae", 58),
        ("large_cutworm", "black_cutworm", 54),
        ("Miridae", "tarnished_plant_bug", 54),
        ("legume_blister_beetle", "blister_beetle", 53),
        ("alfalfa_plant_bug", "Miridae", 52),
        ("Cicadellidae", "Cicadella_viridis", 51),
        ("cabbage_army_worm", "beet_army_worm", 48),
        ("Miridae", "alfalfa_plant_bug", 48),
        ("beet_army_worm", "cabbage_army_worm", 47),
        ("lytta_polita", "blister_beetle", 47),
        ("brown_plant_hopper", "white_backed_plant_hopper", 46),
        ("rice_shell_pest", "rice_leaf_roller", 46),
        ("blister_beetle", "lytta_polita", 42),
        ("white_backed_plant_hopper", "brown_plant_hopper", 41),
        ("black_cutworm", "large_cutworm", 39),
        ("Miridae", "aphids", 39),
        ("bird_cherry-oataphid", "aphids", 36),
        ("aphids", "Miridae", 35),
    ]

    total_instances = sum(count for _, _, count in confused_pairs)
    same_family_instances = 0
    same_order_instances = 0
    unknown_instances = 0

    print("Checking each confused pair against verified taxonomy:\n")
    for true_label, pred_label, count in confused_pairs:
        result = check_pair(true_label, pred_label)

        if not result["known"]:
            tag = "UNKNOWN (not in taxonomy table)"
            unknown_instances += count
        elif result["same_family"]:
            tag = f"SAME FAMILY ({result['true_family']})"
            same_family_instances += count
            same_order_instances += count
        elif result["same_order"]:
            tag = "same order, different family"
            same_order_instances += count
        else:
            tag = "NOT closely related"

        print(f"  {true_label:30s} -> {pred_label:30s}  ({count:3d}x)  {tag}")

    print(f"\nTotal misclassification instances analyzed: {total_instances}")
    print(f"  Same family : {same_family_instances} ({same_family_instances/total_instances:.1%})")
    print(f"  Same order  : {same_order_instances} ({same_order_instances/total_instances:.1%})")
    print(f"  Unknown     : {unknown_instances} ({unknown_instances/total_instances:.1%})")


if __name__ == "__main__":
    main()