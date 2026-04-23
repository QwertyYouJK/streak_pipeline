#!/usr/bin/env python3
"""
Combined ASTRiDE evaluator.

This script uses the newer detection pipeline from evaluate_astride.py:
- Gaussian blur into a temporary FITS file before detection
- tuned ASTRiDE parameters
- minimum length filtering
- connectivity grouping
- merged geometry for linked streak pieces

It also adds the interactive review/reporting flow from evaluate_astride_old.py:
- show each image with detected streak contours
- ask whether each detected streak/group is correct
- ask how many real streaks were missed
- count TP, FP, and FN
- write one CSV row per detected or missed streak

Usage:
    python evaluate_astride_combined.py
    python evaluate_astride_combined.py /path/to/input_dir
    python evaluate_astride_combined.py /path/to/input_dir --output report.csv
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astride import Streak
from scipy.ndimage import gaussian_filter


# ----------------------------
# Detection settings
# ----------------------------
PARAMS = dict(
    remove_bkg="map",
    bkg_box_size=50,
    contour_threshold=2,
    min_points=5,
    shape_cut=0.2,
    area_cut=10,
    radius_dev_cut=0.5,
    connectivity_angle=10.0,
    output_path=None,
)

BLUR_SIGMA = 0.8
MIN_LENGTH = 50
ZERO_ANGLE_TOL_DEG = 0.5
BORDER_CENTER_MARGIN_PX = 1.0


def find_input_files(input_dir: Path):
    files = []
    for pattern in ["**/*.fits", "**/*.fit", "**/*.fts"]:
        files.extend(input_dir.glob(pattern))
    return sorted(files)


def write_blurred_temp_fits(in_fits: Path, sigma: float):
    data, header = fits.getdata(in_fits, header=True, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    fits.writeto(tmp_path, img, header=header, overwrite=True)
    return tmp_path


def image_shape(fpath: Path):
    data = fits.getdata(fpath, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))
    if img.ndim < 2:
        raise ValueError(f"Expected at least a 2D FITS image, got shape {img.shape}")
    return img.shape[-2:]


def run_astride(fpath: Path):
    streak = Streak(str(fpath), **PARAMS)
    streak.detect()

    streak.streaks = [s for s in streak.streaks if s["length"] > MIN_LENGTH]
    return streak


def build_groups(streak):
    """
    Build connected groups of streak indices.
    Treat connectivity as undirected.
    """
    idx_to_streak = {s["index"]: s for s in streak.streaks}
    adj = {idx: set() for idx in idx_to_streak}

    for s in streak.streaks:
        i = s["index"]
        j = s["connectivity"]
        if j != -1 and j in idx_to_streak:
            adj[i].add(j)
            adj[j].add(i)

    groups = []
    seen = set()

    for start in adj:
        if start in seen:
            continue

        stack = [start]
        group = []
        seen.add(start)

        while stack:
            u = stack.pop()
            group.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        groups.append(sorted(group))

    return groups, idx_to_streak


def merged_geometry(group, idx_to_streak):
    """
    Merge a connected group into one center/endpoints using PCA.
    """
    pts = []

    for idx in group:
        s = idx_to_streak[idx]

        x = np.asarray(s.get("x", []), dtype=float)
        y = np.asarray(s.get("y", []), dtype=float)

        if len(x) > 0 and len(y) > 0:
            pts.append(np.column_stack([x, y]))
        else:
            ep1, ep2 = s["extreme_points"]
            pts.append(np.array([[ep1[0], ep1[1]], [ep2[0], ep2[1]]], dtype=float))

    P = np.vstack(pts)

    mean_pt = P.mean(axis=0)
    Q = P - mean_pt
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    direction = vh[0]
    direction = direction / np.linalg.norm(direction)

    t = Q @ direction
    ep1 = mean_pt + t.min() * direction
    ep2 = mean_pt + t.max() * direction
    center = 0.5 * (ep1 + ep2)
    length = np.linalg.norm(ep2 - ep1)

    return center, ep1, ep2, length


def group_records(groups, idx_to_streak):
    records = []

    for group_number, group in enumerate(groups, start=1):
        center, ep1, ep2, merged_length = merged_geometry(group, idx_to_streak)
        raw_lengths = [idx_to_streak[idx]["length"] for idx in group]
        slope_angles = [idx_to_streak[idx]["slope_angle"] for idx in group]

        records.append(
            {
                "group_number": group_number,
                "component_indices": ";".join(str(idx) for idx in group),
                "num_components": len(group),
                "center_x": center[0],
                "center_y": center[1],
                "endpoint1_x": ep1[0],
                "endpoint1_y": ep1[1],
                "endpoint2_x": ep2[0],
                "endpoint2_y": ep2[1],
                "merged_length": merged_length,
                "component_length_sum": float(np.sum(raw_lengths)),
                "mean_slope_angle": float(np.mean(slope_angles)),
            }
        )

    return records


def is_probable_border_artifact(rec, img_shape):
    height, width = img_shape
    max_x = width - 1
    max_y = height - 1

    near_zero_angle = abs(rec["mean_slope_angle"]) <= ZERO_ANGLE_TOL_DEG
    near_border = (
        rec["center_x"] <= BORDER_CENTER_MARGIN_PX
        or rec["center_x"] >= max_x - BORDER_CENTER_MARGIN_PX
        or rec["center_y"] <= BORDER_CENTER_MARGIN_PX
        or rec["center_y"] >= max_y - BORDER_CENTER_MARGIN_PX
    )

    return near_zero_angle and near_border


def filter_border_artifacts(groups, idx_to_streak, img_shape):
    grouped = group_records(groups, idx_to_streak)
    kept_groups = []
    rejected_records = []

    for group, rec in zip(groups, grouped):
        if is_probable_border_artifact(rec, img_shape):
            rejected_records.append(rec)
        else:
            kept_groups.append(group)

    return kept_groups, group_records(kept_groups, idx_to_streak), rejected_records


def show_image_with_groups(fpath: Path, streak, groups):
    data = fits.getdata(fpath, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    vmin, vmax = np.percentile(img, [5, 99.7])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    idx_to_group_number = {}
    for group_number, group in enumerate(groups, start=1):
        for idx in group:
            idx_to_group_number[idx] = group_number

    for s in streak.streaks:
        if s["index"] not in idx_to_group_number:
            continue

        x = np.asarray(s.get("x", []), dtype=float)
        y = np.asarray(s.get("y", []), dtype=float)

        if len(x) > 1 and len(y) > 1:
            x_closed = np.append(x, x[0])
            y_closed = np.append(y, y[0])
            ax.plot(x_closed, y_closed, color="cyan", linewidth=1.5)
            ax.fill(x_closed, y_closed, color="cyan", alpha=0.12)

    # add a dot in the image at pixel location (1704, 928)
    ax.plot(936, 240, "ro", markersize=4)

    for rec in group_records(groups, {s["index"]: s for s in streak.streaks}):
        ax.plot(rec["center_x"], rec["center_y"], "yo", markersize=4)
        ax.text(
            rec["center_x"],
            rec["center_y"],
            str(rec["group_number"]),
            color="yellow",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    ax.set_title(f"{fpath.name} - detected groups: {len(groups)}")
    plt.tight_layout()
    plt.show()


def show_image_only(fpath: Path):
    data = fits.getdata(fpath, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    vmin, vmax = np.percentile(img, [5, 99.7])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{fpath.name} - no kept detections")
    plt.tight_layout()
    plt.show()


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


def ask_nonnegative_int(prompt: str) -> int:
    while True:
        ans = input(prompt + " [0, 1, 2, ...]: ").strip()
        try:
            value = int(ans)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value >= 0:
            return value

        print("Please enter 0 or a positive whole number.")


def fmt_float(value):
    if value == "":
        return ""
    return f"{value:.3f}"


def add_detected_row(
    records, fpath, image_index, total_images, num_raw, num_groups, rec, outcome
):
    records.append(
        {
            "record_type": "detected_streak",
            "outcome": outcome,
            "filename": fpath.name,
            "filepath": str(fpath),
            "image_index": image_index,
            "total_images": total_images,
            "num_raw_detections": num_raw,
            "num_detected_groups": num_groups,
            "group_number": rec["group_number"],
            "component_indices": rec["component_indices"],
            "num_components": rec["num_components"],
            "center_x": fmt_float(rec["center_x"]),
            "center_y": fmt_float(rec["center_y"]),
            "endpoint1_x": fmt_float(rec["endpoint1_x"]),
            "endpoint1_y": fmt_float(rec["endpoint1_y"]),
            "endpoint2_x": fmt_float(rec["endpoint2_x"]),
            "endpoint2_y": fmt_float(rec["endpoint2_y"]),
            "merged_length": fmt_float(rec["merged_length"]),
            "component_length_sum": fmt_float(rec["component_length_sum"]),
            "mean_slope_angle": fmt_float(rec["mean_slope_angle"]),
            "human_correct_detection": 1 if outcome == "TP" else 0,
            "missed_streak_number": "",
            "notes": "",
        }
    )


def add_missed_rows(
    records, fpath, image_index, total_images, num_raw, num_groups, missed_count
):
    for missed_number in range(1, missed_count + 1):
        records.append(
            {
                "record_type": "missed_streak",
                "outcome": "FN",
                "filename": fpath.name,
                "filepath": str(fpath),
                "image_index": image_index,
                "total_images": total_images,
                "num_raw_detections": num_raw,
                "num_detected_groups": num_groups,
                "group_number": "",
                "component_indices": "",
                "num_components": "",
                "center_x": "",
                "center_y": "",
                "endpoint1_x": "",
                "endpoint1_y": "",
                "endpoint2_x": "",
                "endpoint2_y": "",
                "merged_length": "",
                "component_length_sum": "",
                "mean_slope_angle": "",
                "human_correct_detection": "",
                "missed_streak_number": missed_number,
                "notes": "Human reported a real streak that ASTRiDE did not detect.",
            }
        )


def write_report(records, out_csv: Path):
    fieldnames = [
        "record_type",
        "outcome",
        "filename",
        "filepath",
        "image_index",
        "total_images",
        "num_raw_detections",
        "num_detected_groups",
        "group_number",
        "component_indices",
        "num_components",
        "center_x",
        "center_y",
        "endpoint1_x",
        "endpoint1_y",
        "endpoint2_x",
        "endpoint2_y",
        "merged_length",
        "component_length_sum",
        "mean_slope_angle",
        "human_correct_detection",
        "missed_streak_number",
        "notes",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(tp, fp, fn, reviewed_images, no_streak_images):
    total_streak_decisions = tp + fp + fn

    print("\n" + "#" * 60)
    print("SUMMARY REPORT")
    print("#" * 60)
    print(f"Images reviewed                         : {reviewed_images}")
    print(f"Images with no detected or missed streak: {no_streak_images}")
    print(f"Total streak decisions                  : {total_streak_decisions}")
    print(f"  TP - correct detections               : {tp}")
    print(f"  FP - incorrect detections             : {fp}")
    print(f"  FN - missed real streaks              : {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision                               : {precision:.3f}")
    else:
        print("Precision                               : n/a")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall                                  : {recall:.3f}")
    else:
        print("Recall                                  : n/a")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASTRiDE and interactively label TP/FP/FN streaks."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=Path(__file__).parent / "input",
        type=Path,
        help="Directory containing FITS files. Defaults to ./input next to this script.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=Path(__file__).parent / "astride_evaluation_report_combined.csv",
        type=Path,
        help="CSV output path. Defaults to astride_evaluation_report_combined.csv.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Do not open matplotlib image windows. Useful for testing the script flow.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    out_csv = args.output.expanduser().resolve()

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    fits_files = find_input_files(input_dir)
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(fits_files)} FITS files in {input_dir}")
    print(f"Gaussian blur sigma: {BLUR_SIGMA}")
    print(f"Minimum kept streak length: {MIN_LENGTH}")
    print(
        "Border artifact filter: "
        f"|mean_angle| <= {ZERO_ANGLE_TOL_DEG} deg and "
        f"center within {BORDER_CENTER_MARGIN_PX} px of an image edge"
    )
    print(f"Report will be written to: {out_csv}")

    tp = 0
    fp = 0
    fn = 0
    reviewed_images = 0
    no_streak_images = 0
    records = []

    try:
        for image_index, fpath in enumerate(fits_files, start=1):
            reviewed_images += 1
            print("\n" + "=" * 70)
            print(f"[{image_index}/{len(fits_files)}] Processing: {fpath}")

            temp_fits = write_blurred_temp_fits(fpath, BLUR_SIGMA)
            try:
                streak = run_astride(temp_fits)
            finally:
                temp_fits.unlink(missing_ok=True)

            img_shape = image_shape(fpath)
            groups, idx_to_streak = build_groups(streak)
            raw_group_count = len(groups)
            groups, grouped, rejected_grouped = filter_border_artifacts(
                groups, idx_to_streak, img_shape
            )

            print(f"Kept raw ASTRiDE components: {len(streak.streaks)}")
            print(
                f"Connected candidate streak groups before border filter: {raw_group_count}"
            )
            print(
                f"Connected candidate streak groups after border filter: {len(grouped)}"
            )

            for rec in rejected_grouped:
                print(
                    f"  rejected probable border artifact group #{rec['group_number']}: "
                    f"components={rec['component_indices']}, "
                    f"center=({rec['center_x']:.2f}, {rec['center_y']:.2f}), "
                    f"mean_angle={rec['mean_slope_angle']:.3f}"
                )

            for rec in grouped:
                print(
                    f"  group #{rec['group_number']}: "
                    f"components={rec['component_indices']}, "
                    f"center=({rec['center_x']:.2f}, {rec['center_y']:.2f}), "
                    f"length={rec['merged_length']:.2f}, "
                    f"mean_angle={rec['mean_slope_angle']:.2f}"
                )

            if not args.skip_display:
                if grouped:
                    show_image_with_groups(fpath, streak, groups)
                else:
                    show_image_only(fpath)

            for rec in grouped:
                is_correct = ask_yes_no(
                    f"Is detected group #{rec['group_number']} a correct streak detection?"
                )
                outcome = "TP" if is_correct else "FP"

                if is_correct:
                    tp += 1
                else:
                    fp += 1

                add_detected_row(
                    records,
                    fpath,
                    image_index,
                    len(fits_files),
                    len(streak.streaks),
                    len(grouped),
                    rec,
                    outcome,
                )

            missed_count = ask_nonnegative_int(
                "How many real streaks in this image did ASTRiDE miss?"
            )
            fn += missed_count
            add_missed_rows(
                records,
                fpath,
                image_index,
                len(fits_files),
                len(streak.streaks),
                len(grouped),
                missed_count,
            )

            if not grouped and missed_count == 0:
                no_streak_images += 1

            print(f"Image outcome: TP={tp}, FP={fp}, FN={fn} so far")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing partial report...")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_report(records, out_csv)
    print_summary(tp, fp, fn, reviewed_images, no_streak_images)
    print(f"\nCSV report written to: {out_csv}")


if __name__ == "__main__":
    main()
