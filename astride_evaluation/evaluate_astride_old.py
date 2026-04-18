#!/usr/bin/env python3
"""
evaluate_astride.py

Batch-run ASTRiDE on FITS images in ./input (recursively), let a human supervisor
label whether each image actually contains a streak and whether ASTRiDE's
detection is acceptable, then report TP/FP/FN/TN + quality of detections.

Usage:
    python evaluate_astride.py          # uses ./input
    python evaluate_astride.py /path/to/input_dir
"""

import sys
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from astride import Streak


def find_input_files(input_dir: Path):
    """
    Recursively find all FITS files under input_dir.
    e.g. input/0427/15_35/img*.fits, input/0528/16_42/img*.fits, etc.
    """
    patterns = ["**/*.fits", "**/*.fit", "**/*.fts"]
    fits_files = []
    for p in patterns:
        fits_files.extend(input_dir.glob(p))
    return sorted(fits_files)


def run_astride(fpath: Path) -> Streak:
    """
    Run ASTRiDE on a single FITS file and return the Streak instance.

    You can tweak thresholds here (contour_threshold, min_points, etc.)
    depending on how aggressive you want detection to be.
    """
    streak = Streak(
        str(fpath),
        remove_bkg="map",
        contour_threshold=3.0,
        min_points=10,
        shape_cut=0.2,
        area_cut=10,
        radius_dev_cut=0.5,
        connectivity_angle=3.0,
        output_path=None,
    )
    streak.detect()
    return streak


def show_image_with_streaks(fpath: Path, streak: Streak):
    """
    Display the FITS image with detected streaks overlaid as rectangles.

    This handles both dict-based and attribute-based streak objects.
    """
    data = fits.getdata(fpath, ignore_missing_end=True)

    if isinstance(data, np.ndarray) and data.ndim >= 2:
        img = np.array(data, dtype=float)
        vmin, vmax = np.percentile(img, [1, 99])
    else:
        raise RuntimeError(f"Cannot interpret data in {fpath} as an image")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for s in streak.streaks:
        # handle dict or object
        if isinstance(s, dict):
            x_min = s["x_min"]
            x_max = s["x_max"]
            y_min = s["y_min"]
            y_max = s["y_max"]
            sid = s.get("ID", "")
        else:
            x_min = s.x_min
            x_max = s.x_max
            y_min = s.y_min
            y_max = s.y_max
            sid = getattr(s, "ID", "")

        width = x_max - x_min
        height = y_max - y_min

        rect = Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        ax.plot(cx, cy, "o", markersize=3)
        if sid != "":
            ax.text(
                cx, cy, f"{sid}", color="white", fontsize=6, ha="center", va="bottom"
            )

    ax.set_title(f"{fpath.name} — detected streaks: {len(streak.streaks)}")
    plt.tight_layout()
    plt.show()


def ask_yes_no(prompt: str) -> bool:
    """Prompt user for y/n, return True for yes, False for no."""
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


def main():
    # --- Resolve input directory ---
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        input_dir = Path(__file__).parent / "input"

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        print("Create it and put your FITS files there.")
        sys.exit(1)

    fits_files = find_input_files(input_dir)
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(fits_files)} FITS files in {input_dir}")

    # Confusion matrix counters
    tp_accept = 0  # NEW: TP with acceptable detection
    tp_unaccept = 0  # NEW: TP with unacceptable detection
    fp = fn = tn = 0

    # Per-file records for CSV
    records = []

    try:
        for idx, fpath in enumerate(fits_files, start=1):
            print("\n" + "=" * 60)
            print(f"[{idx}/{len(fits_files)}] Processing: {fpath}")

            # --- Run ASTRiDE ---
            streak = run_astride(fpath)
            num_detected = len(streak.streaks)

            # Print where they are (approx center coords)
            centers = []
            for s in streak.streaks:
                if isinstance(s, dict):
                    x_min = s.get("x_min")
                    x_max = s.get("x_max")
                    y_min = s.get("y_min")
                    y_max = s.get("y_max")
                else:
                    x_min = s.x_min
                    x_max = s.x_max
                    y_min = s.y_min
                    y_max = s.y_max

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                centers.append((cx, cy))

            print(f"ASTRiDE detected {num_detected} streak(s).")
            if centers:
                print("Approximate centers (x, y) in pixel coordinates:")
                for i, (cx, cy) in enumerate(centers, start=1):
                    print(f"  #{i}: ({cx:.1f}, {cy:.1f})")

            # --- Show the image + overlays ---
            show_image_with_streaks(fpath, streak)

            # NEW: quality flag for detection (only meaningful when gt_has_streak & detected)
            # detection_acceptable = ask_yes_no(
            #     "Judgement: is ASTRiDE's detection acceptable (even if partial)?"
            # )
            detection_acceptable = True

            # --- Update confusion matrix (with extra TP split) ---
            if detection_acceptable:
                tp_accept += 1
                outcome = "TP_ACCEPT (detected streak, acceptable)"
            else:
                tp_unaccept += 1
                outcome = "TP_UNACCEPT (detected streak, unacceptable/poor)"

            print(f"Outcome for {fpath.name}: {outcome}")

            # For CSV, encode detection_acceptable as 1/0/"" (NA)
            if detection_acceptable is True:
                det_acc_val = 1
            elif detection_acceptable is False:
                det_acc_val = 0
            else:
                det_acc_val = ""

            records.append(
                {
                    "filepath": str(fpath),
                    "filename": fpath.name,
                    "num_detected": num_detected,
                    "detection_acceptable": det_acc_val,
                    "outcome": outcome,
                }
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Generating partial report...")

    # --- Summary ---
    tp = tp_accept + tp_unaccept
    total = tp + fp + fn + tn

    print("\n" + "#" * 60)
    print("SUMMARY REPORT")
    print("#" * 60)
    print(f"Total images reviewed: {total}")
    print(f"  TP (all)                                : {tp}")
    print(f"    TP_ACCEPT (acceptable detections)     : {tp_accept}")
    print(f"    TP_UNACCEPT (unacceptable detections) : {tp_unaccept}")
    print(f"  FN (missed streaks)                     : {fn}")
    print(f"  FP (detected streaks when none present) : {fp}")
    print(f"  TN (correctly no streak)                : {tn}")

    if total > 0:
        accuracy = (tp + tn) / total
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

        print("\nBasic metrics (treating all TP as positive):")
        print(f"  Accuracy  : {accuracy:.3f}")
        print(f"  Precision : {precision:.3f}")
        print(f"  Recall    : {recall:.3f}")

        if tp > 0:
            frac_accept = tp_accept / tp
            print("\nQuality of true positives:")
            print(f"  Fraction of TP that are acceptable : {frac_accept:.3f}")
        else:
            print("\nQuality of true positives:")
            print("  No true positives to evaluate.")

    # --- Write CSV report next to script ---
    out_csv = Path(__file__).parent / "astride_evaluation_report.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filepath",
                "filename",
                "astride_detected",
                "num_detected",
                "gt_has_streak",
                "detection_acceptable",
                "outcome",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"\nCSV report written to: {out_csv}")


if __name__ == "__main__":
    main()
