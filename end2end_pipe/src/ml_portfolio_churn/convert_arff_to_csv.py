#!/usr/bin/env python3
"""
ARFF to CSV Converter

Prefers `liac-arff` for robust parsing (quoted names, nominal types, escaped values),
with a manual parser fallback when `liac-arff` is unavailable.
"""

import csv
import logging
import re
import sys
from pathlib import Path


def convert_arff_to_csv(arff_path: str, csv_path: str = None):
    """Convert ARFF file to CSV format using liac-arff if available."""

    arff_file = Path(arff_path)
    if not arff_file.exists():
        logging.getLogger(__name__).error(f"ARFF file not found: {arff_path}")
        return False

    if csv_path is None:
        csv_path = arff_file.with_suffix(".csv")

    logging.getLogger(__name__).info(f"Converting {arff_path} to {csv_path}")

    # Try liac-arff first
    try:
        import arff  # liac-arff

        with open(arff_path, "r", encoding="utf-8") as f:
            dataset = arff.load(f)
        attributes = [name for name, _type in dataset.get("attributes", [])]
        data = dataset.get("data", [])
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(attributes)
            for row in data:
                writer.writerow(row)
        logging.getLogger(__name__).info(f"Successfully converted to {csv_path} using liac-arff")
        return True

    except Exception as e_liac:
        logging.getLogger(__name__).warning(
            f"liac-arff unavailable or failed: {e_liac}. Falling back to manual parser..."
        )

    # Manual fallback parser
    try:
        with open(arff_path, "r", encoding="utf-8") as arff_file_obj:
            lines = arff_file_obj.readlines()

        # Find attribute names
        attributes = []
        data_start = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            lower = line_stripped.lower()
            if lower.startswith("@attribute"):
                # Support quoted or bare attribute names
                attr_match = re.match(
                    r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))",
                    line_stripped,
                    re.IGNORECASE,
                )
                if attr_match:
                    attr_name = next(g for g in attr_match.groups() if g is not None)
                    attributes.append(attr_name)
            elif lower.startswith("@data"):
                data_start = i + 1
                break

        logging.getLogger(__name__).info(f"Found {len(attributes)} attributes")
        logging.getLogger(__name__).debug(f"Attributes: {attributes}")

        # Write CSV file
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            # Write header
            writer.writerow(attributes)

            # Write data rows
            row_count = 0
            for line in lines[data_start:]:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("%"):
                    continue

                # Parse data using csv with support for both quote styles
                parsed = None
                for quote in ("'", '"'):
                    try:
                        reader = csv.reader([line], quotechar=quote, skipinitialspace=True)
                        parsed = next(reader)
                        break
                    except Exception:
                        parsed = None
                if parsed is None:
                    parsed = [p.strip().strip("'\"") for p in line.split(",")]
                if parsed:
                    writer.writerow(parsed)
                    row_count += 1

            logging.getLogger(__name__).info(f"Converted {row_count} data rows")

        logging.getLogger(__name__).info(f"Successfully converted to {csv_path}")
        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"Error converting file: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error(
            "Usage: python convert_arff_to_csv.py <arff_file> [output_csv]"
        )
        logging.getLogger(__name__).error(
            "Example: python convert_arff_to_csv.py data/dataset.arff data/dataset.csv"
        )
        sys.exit(1)

    arff_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_arff_to_csv(arff_path, csv_path)
    sys.exit(0 if success else 1)
