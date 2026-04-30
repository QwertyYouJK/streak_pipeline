# Python script to convert ra_center and dec_center columns from h:m:s / d:m:s format into degrees in CSV files.
# Usage: python csv_radec_convert.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def parse_sexagesimal(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.strip().split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Expected a value in colon-separated sexagesimal format, got {value!r}"
        )

    if len(parts) == 2:
        parts.append("0")

    return float(parts[0]), float(parts[1]), float(parts[2])


def ra_hms_to_degrees(value: str) -> float:
    hours, minutes, seconds = parse_sexagesimal(value)
    total_hours = hours + minutes / 60.0 + seconds / 3600.0
    return total_hours * 15.0


def dec_dms_to_degrees(value: str) -> float:
    degrees, arcminutes, arcseconds = parse_sexagesimal(value)
    sign = -1.0 if degrees < 0 else 1.0
    total_degrees = abs(degrees) + arcminutes / 60.0 + arcseconds / 3600.0
    return sign * total_degrees


def format_degrees(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def iter_input_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(path).resolve() for path in paths]

    return sorted(DEFAULT_INPUT_DIR.glob("*.csv"))


def convert_file(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_degrees.csv"

    with input_path.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        missing_columns = {"ra_center", "dec_center"} - set(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise KeyError(
                f"{input_path.name} is missing required column(s): {missing}"
            )

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["ra_deg", "dec_deg"])

            for row in reader:
                ra_text = (row.get("ra_center") or "").strip()
                dec_text = (row.get("dec_center") or "").strip()
                if not ra_text and not dec_text:
                    continue
                if not ra_text or not dec_text:
                    raise ValueError(
                        f"{input_path.name} contains a row with a missing ra_center or dec_center value"
                    )

                ra_deg = ra_hms_to_degrees(ra_text)
                dec_deg = dec_dms_to_degrees(dec_text)
                writer.writerow([format_degrees(ra_deg), format_degrees(dec_deg)])

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ra_center/dec_center columns from h:m:s / d:m:s into degrees."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Optional CSV file paths. If omitted, every CSV in the input folder is processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where converted CSV files will be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_files = iter_input_files(args.inputs)
    if not input_files:
        parser.error(f"No CSV files found in {DEFAULT_INPUT_DIR}")

    output_dir = Path(args.output_dir).resolve()
    for input_file in input_files:
        output_path = convert_file(input_file, output_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
