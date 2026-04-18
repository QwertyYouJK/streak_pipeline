#!/usr/bin/env python3
"""
Simple ASTRiDE batch runner.

What it does:
- finds FITS files in ./input
- optionally blurs each image slightly before detection
- runs ASTRiDE with one preset
- rejects short detections
- groups linked streak pieces using connectivity
- prints merged center/endpoints for each group
- shows the original image with detected contours
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astride import Streak


# ----------------------------
# Settings
# ----------------------------
PARAMS = dict(
    remove_bkg="map",
    bkg_box_size=50,
    contour_threshold=0.9,
    min_points=10,
    shape_cut=0.2,
    area_cut=10,
    radius_dev_cut=0.45,
    connectivity_angle=10.0,
    output_path=None,
)

BLUR_SIGMA = 0.8
MIN_LENGTH = 50


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


def run_astride(fpath: Path):
    streak = Streak(str(fpath), **PARAMS)
    streak.detect()

    # keep only long enough detections
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


def show_result(fpath: Path, streak):
    data = fits.getdata(fpath, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    vmin, vmax = np.percentile(img, [5, 99.7])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    for s in streak.streaks:
        x = np.asarray(s.get("x", []), dtype=float)
        y = np.asarray(s.get("y", []), dtype=float)

        if len(x) > 1 and len(y) > 1:
            x_closed = np.append(x, x[0])
            y_closed = np.append(y, y[0])
            ax.plot(x_closed, y_closed, color="cyan", linewidth=1.5)
            ax.fill(x_closed, y_closed, color="cyan", alpha=0.12)

            cx = np.mean(x)
            cy = np.mean(y)
            ax.plot(cx, cy, "yo", markersize=3)

    ax.set_title(f"{fpath.name} — detected streaks: {len(streak.streaks)}")
    plt.tight_layout()
    plt.show()


def show_fit(fpath: Path):
    data = fits.getdata(fpath, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    vmin, vmax = np.percentile(img, [5, 99.7])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    ax.set_title(f"{fpath.name}")
    plt.tight_layout()
    plt.show()


def main():
    input_dir = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else Path(__file__).parent / "input"
    )

    fits_files = find_input_files(input_dir)
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(fits_files)} FITS files in {input_dir}")

    try:
        for i, fpath in enumerate(fits_files, start=1):
            print("\n" + "=" * 70)
            print(f"[{i}/{len(fits_files)}] Processing: {fpath}")

            # streak = run_astride(fpath)
            temp_fits = write_blurred_temp_fits(fpath, BLUR_SIGMA)

            # show_fit(temp_fits)

            try:
                streak = run_astride(temp_fits)
            finally:
                temp_fits.unlink(missing_ok=True)

            if not streak.streaks:
                print("No streaks kept after filtering.")
                show_result(fpath, streak)
                continue

            print(f"Kept {len(streak.streaks)} streak(s):")
            for s in streak.streaks:
                print(
                    f"  index={s['index']}, "
                    f"connectivity={s['connectivity']}, "
                    f"slope_angle={s['slope_angle']:.2f}, "
                    f"length={s['length']:.2f}"
                )

            groups, idx_to_streak = build_groups(streak)

            for group in groups:
                center, ep1, ep2, length = merged_geometry(group, idx_to_streak)
                print(f"group: {group}")
                print(f"  center   : ({center[0]:.2f}, {center[1]:.2f})")
                print(f"  endpoint1: ({ep1[0]:.2f}, {ep1[1]:.2f})")
                print(f"  endpoint2: ({ep2[0]:.2f}, {ep2[1]:.2f})")
                print(f"  length   : {length:.2f}")

            show_result(fpath, streak)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
