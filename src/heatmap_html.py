"""
heatmap_html.py

Generates an interactive HTML heatmap showing trap locations and pest
detections across the field. Built with Folium (a Python wrapper for
Leaflet.js) so the output is a standalone HTML file that works in any
browser with no server needed.

Features:
  - Interactive map with zoom and pan
  - Circle markers sized by insect count
  - Color coded by pest density (green to red)
  - Hover tooltip showing trap ID and insect count
  - Click popup showing full species breakdown per trap
  - Summary panel embedded in the map

Usage:
    python src/heatmap_html.py --demo
    python src/heatmap_html.py --input results/detection_log.json --output results/
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
from datetime import datetime


def simulate_detection_log(n_traps=12):
    """
    Generate simulated detection data for testing.

    Purpose:
        Allows the HTML heatmap to be built and tested without real
        Amiga GPS data. Real coordinates and detections from the Amiga
        pipeline will replace this during field validation.

    Args:
        n_traps (int): Number of trap locations to simulate.

    Returns:
        list: List of trap result dicts with GPS coordinates and detections.
    """
    random.seed(42)
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


def get_marker_color(count):
    """
    Map insect count to a color for the map marker.

    Purpose:
        Provides an intuitive color scale so farmers can immediately
        see which traps have high pest pressure without reading numbers.

    Args:
        count (int): Number of insects detected at this trap.

    Returns:
        str: A color string compatible with Folium CircleMarker.
    """
    if count == 0:
        return "#2D6A4F"   # dark green - clear
    elif count <= 2:
        return "#95D5B2"   # light green - low
    elif count <= 4:
        return "#F4A261"   # orange - medium
    elif count <= 6:
        return "#E76F51"   # dark orange - high
    else:
        return "#C1121F"   # red - very high


def generate_html_heatmap(log, output_dir):
    """
    Generate a standalone interactive HTML heatmap using Folium.

    Purpose:
        Creates a Leaflet.js map with one circle marker per trap.
        Markers are sized and colored by insect count. Hovering shows
        a tooltip with trap ID and count. Clicking opens a popup with
        the full species breakdown for that trap. A summary panel is
        embedded in the top-right corner of the map.

    Args:
        log        (list): List of trap result dicts with GPS coordinates
                           and detection data.
        output_dir (Path): Folder to save the HTML file.

    Returns:
        Path: Path to the saved HTML file.
    """
    try:
        import folium
    except ImportError:
        print("Installing folium...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "folium", "-q"])
        import folium

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center_lat = sum(t["latitude"] for t in log) / len(log)
    center_lon = sum(t["longitude"] for t in log) / len(log)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="OpenStreetMap"
    )

    for trap in log:
        count = trap["insect_count"]
        color = get_marker_color(count)
        radius = max(8, count * 4)

        species_counter = Counter(d["species"] for d in trap.get("detections", []))

        if species_counter:
            species_rows = "".join([
                f"<tr><td style='padding:3px 8px;'>{s.replace('_',' ')}</td>"
                f"<td style='padding:3px 8px; text-align:center;'><b>{c}</b></td></tr>"
                for s, c in species_counter.most_common()
            ])
            species_table = f"""
            <table style='border-collapse:collapse; width:100%; margin-top:6px;'>
              <tr style='background:#2D6A4F; color:white;'>
                <th style='padding:4px 8px; text-align:left;'>Species</th>
                <th style='padding:4px 8px;'>Count</th>
              </tr>
              {species_rows}
            </table>"""
        else:
            species_table = "<p style='color:#666; margin-top:6px;'>No insects detected</p>"

        popup_html = f"""
        <div style='font-family:Arial,sans-serif; min-width:200px;'>
          <h4 style='margin:0 0 4px 0; color:#1B3A2D;'>{trap["trap_id"].upper()}</h4>
          <p style='margin:2px 0; font-size:13px;'>
            <b>Insects detected:</b> {count}
          </p>
          <p style='margin:2px 0; font-size:11px; color:#666;'>
            {trap["latitude"]:.6f}, {trap["longitude"]:.6f}
          </p>
          {species_table}
        </div>"""

        tooltip_text = f"{trap['trap_id']}: {count} insect{'s' if count != 1 else ''}"

        folium.CircleMarker(
            location=[trap["latitude"], trap["longitude"]],
            radius=radius,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(tooltip_text, sticky=True),
            popup=folium.Popup(popup_html, max_width=280)
        ).add_to(m)

    all_species = []
    for t in log:
        for d in t.get("detections", []):
            all_species.append(d["species"])

    total_insects = sum(t["insect_count"] for t in log)
    total_traps = len(log)
    positive_traps = sum(1 for t in log if t["insect_count"] > 0)

    if all_species:
        top_species = Counter(all_species).most_common(5)
        top_rows = "".join([
            f"<tr><td style='padding:2px 6px;font-size:12px;'>"
            f"{s.replace('_',' ')}</td>"
            f"<td style='padding:2px 6px;font-size:12px;text-align:center;'>"
            f"<b>{c}</b></td></tr>"
            for s, c in top_rows
        ]) if False else "".join([
            f"<tr><td style='padding:2px 6px;font-size:12px;'>"
            f"{s.replace('_',' ')}</td>"
            f"<td style='padding:2px 6px;font-size:12px;text-align:center;'>"
            f"<b>{c}</b></td></tr>"
            for s, c in top_species
        ])
    else:
        top_rows = "<tr><td colspan='2' style='padding:4px;color:#666;'>None</td></tr>"

    legend_html = f"""
    <div style='position:fixed; top:10px; right:10px; z-index:1000;
                background:white; padding:14px; border-radius:8px;
                border:1px solid #ccc; font-family:Arial,sans-serif;
                box-shadow:2px 2px 6px rgba(0,0,0,0.15); min-width:200px;'>
      <h4 style='margin:0 0 8px 0; color:#1B3A2D; font-size:14px;'>
        Pest Monitoring Report
      </h4>
      <p style='margin:2px 0; font-size:12px;'>
        <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
      </p>
      <p style='margin:2px 0; font-size:12px;'>
        <b>Traps scanned:</b> {total_traps}
      </p>
      <p style='margin:2px 0; font-size:12px;'>
        <b>Traps positive:</b> {positive_traps}/{total_traps}
      </p>
      <p style='margin:2px 0; font-size:12px;'>
        <b>Total insects:</b> {total_insects}
      </p>
      <hr style='margin:8px 0; border:none; border-top:1px solid #eee;'>
      <p style='margin:4px 0; font-size:12px; font-weight:bold;'>Top species:</p>
      <table style='border-collapse:collapse; width:100%;'>
        {top_rows}
      </table>
      <hr style='margin:8px 0; border:none; border-top:1px solid #eee;'>
      <p style='margin:4px 0; font-size:11px; color:#666;'>
        Click any marker for species breakdown
      </p>
      <div style='margin-top:6px;'>
        <span style='background:#2D6A4F;color:white;padding:2px 6px;
                     border-radius:3px;font-size:10px;'>Clear</span>
        <span style='background:#95D5B2;padding:2px 6px;
                     border-radius:3px;font-size:10px;'>Low</span>
        <span style='background:#F4A261;padding:2px 6px;
                     border-radius:3px;font-size:10px;'>Med</span>
        <span style='background:#C1121F;color:white;padding:2px 6px;
                     border-radius:3px;font-size:10px;'>High</span>
      </div>
    </div>"""

    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = output_dir / "pest_heatmap.html"
    m.save(str(out_path))
    print(f"Interactive HTML heatmap saved to {out_path}")
    return out_path


def main():
    """
    Main entry point. Generates the interactive HTML heatmap.

    Args:
        None (reads from command line arguments)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML pest heatmap")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to detection log JSON from pipeline")
    parser.add_argument("--output", type=str, default="results",
                        help="Folder to save HTML output")
    parser.add_argument("--demo", action="store_true",
                        help="Run with simulated data")
    args = parser.parse_args()

    if args.demo or args.input is None:
        print("Running with simulated GPS data (demo mode)...")
        log = simulate_detection_log(n_traps=12)
    else:
        with open(args.input) as f:
            data = json.load(f)
        log = data.get("results", data)

    generate_html_heatmap(log, args.output)


if __name__ == "__main__":
    main()