#!/usr/bin/env python3
"""Add device column to CSV files in results/ - device extracted from filename."""
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

for fpath in RESULTS_DIR.glob("*.csv"):
    # Skip combined_overall.csv - it doesn't have device in the name
    if fpath.name == "combined_overall.csv":
        continue

    # Extract device from filename: model_algo_type_DEVICE.csv -> DEVICE
    stem = fpath.stem  # e.g. densenet121_fft_tiling_layers_h100
    parts = stem.split("_")
    if len(parts) < 2:
        continue
    device = parts[-1]  # h100, l40s, v100

    rows = []
    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)
        for row in reader:
            rows.append(row)

    # Add 'device' column to header and to every data row
    rows[0].append("device")
    for i in range(1, len(rows)):
        rows[i].append(device)

    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Updated {fpath.name} with device={device}")
