# streak_pipeline

Pipeline for plate-solving FITS images with ASTAP, detecting streaks with a modified ASTRiDE, and writing image/streak coordinates to a CSV.

## Prerequisites
- Python 3
- [ASTAP](https://www.hnsky.org/astap.htm) installed and available on `PATH`
- ASTAP's D80 database installed

## Install
```bash
git clone <repo-url>
cd streak_pipeline
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

This installs the Python dependencies used by the pipeline, including the modified ASTRiDE bundled in this repository.

## Input

Put your FITS files in `solve_pipeline/input/`.

The pipeline searches `input/` recursively for `.fits`, `.fit`, and `.fts`.

## Run

```bash
cd solve_pipeline
python3 pipeline.py report.csv
```

The output filename must end in `.csv`.

If you pass only a filename such as `report.csv`, the pipeline writes it to `solve_pipeline/out/report.csv`. Note that it will overwrite file with the same name.

## Output

The CSV includes:
- `file`
- `astap_status`
- `image_center_ra_deg`
- `image_center_dec_deg`
- `streak_center_x_px`
- `streak_center_y_px`
- `streak_center_ra_deg`
- `streak_center_dec_deg`

The pipeline writes one row per kept streak group. If no streak is kept for an image, it still writes one row with blank streak-coordinate fields.

## Notes

- This pipeline was written and tested on Linux, specifically Ubuntu 22.04. Windows may work but has not been tested yet.
- ASTAP is required separately and is not installed by `pip`.

## Troubleshooting

- `ASTAP not found on PATH`: make sure ASTAP is installed and the `astap` command works in your terminal.
- `No FITS files found`: make sure your files are inside `solve_pipeline/input/`.
