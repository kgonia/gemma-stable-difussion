#!/usr/bin/env python3
"""Create a reproducible aggregate EXIF geometry audit for an image tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


AUDIT_SCHEMA_VERSION = 2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
SWAPPED_ORIENTATIONS = {5, 6, 7, 8}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value or "").strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sensor_key(make: str, model: str) -> str:
    return f"{make.strip().lower()}|{model.strip().lower()}"


def _load_sensor_table(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    table = json.loads(raw)
    if table.get("schema_version") != 1 or not isinstance(table.get("models"), dict):
        raise ValueError(f"Unsupported sensor table schema: {path}")
    required = {
        "sensor_width_mm", "sensor_height_mm", "source_url", "source_title",
        "specification_id", "retrieved_date",
    }
    for model, record in table["models"].items():
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(
                f"Sensor table entry {model!r} lacks provenance: {missing}")
        if not str(record["source_url"]).startswith("https://"):
            raise ValueError(f"Sensor table entry {model!r} needs an HTTPS source")
    return table["models"], hashlib.sha256(raw).hexdigest()


def _dimension_pair(exif: dict[str, Any]) -> tuple[int, int] | None:
    width = _number(exif.get("ExifImageWidth") or exif.get("PixelXDimension"))
    height = _number(exif.get("ExifImageHeight") or exif.get("PixelYDimension"))
    if width and height:
        return int(width), int(height)
    width = _number(exif.get("ImageWidth"))
    height = _number(exif.get("ImageLength"))
    return (int(width), int(height)) if width and height else None


def _derived_sensor(exif: dict[str, Any]) -> tuple[float, float] | None:
    dimensions = _dimension_pair(exif)
    x_resolution = _number(exif.get("FocalPlaneXResolution"))
    y_resolution = _number(exif.get("FocalPlaneYResolution"))
    unit = UNIT_MM.get(int(_number(exif.get("FocalPlaneResolutionUnit")) or 0))
    if not dimensions or not x_resolution or not y_resolution or unit is None:
        return None
    return dimensions[0] / x_resolution * unit, dimensions[1] / y_resolution * unit


def _dimension_status(
    decoded_size: tuple[int, int], exif: dict[str, Any],
) -> str:
    dimensions = _dimension_pair(exif)
    if dimensions is None:
        return "missing"
    if decoded_size == dimensions:
        return "exact"
    orientation = int(_number(exif.get("Orientation")) or 1)
    if decoded_size == dimensions[::-1]:
        return (
            "orientation_corrected"
            if orientation in SWAPPED_ORIENTATIONS else "unexplained_swap"
        )
    return "mismatch"


def classify_geometry_record(
    decoded_size: tuple[int, int], exif: dict[str, Any], crop_state: str,
    sensor_table: dict,
) -> dict:
    """Classify strict geometry eligibility with explicit rejection reasons."""
    dimension_status = _dimension_status(decoded_size, exif)
    derived = _derived_sensor(exif)
    make = _text(exif.get("Make"))
    model = _text(exif.get("Model"))
    known = sensor_table.get(_sensor_key(make, model))
    if derived is None:
        sensor_status = "not_derivable"
    elif known is None:
        sensor_status = "unknown_model"
    else:
        derived_diagonal = math.hypot(*derived)
        known_diagonal = math.hypot(
            float(known["sensor_width_mm"]),
            float(known["sensor_height_mm"]),
        )
        sensor_status = (
            "within_10_percent"
            if abs(derived_diagonal / known_diagonal - 1.0) <= 0.10
            else "outlier"
        )
    focal_present = _number(exif.get("FocalLength")) is not None
    reasons = []
    if dimension_status not in {"exact", "orientation_corrected"}:
        reasons.append(f"dimension_{dimension_status}")
    if crop_state != "false":
        reasons.append(f"crop_{crop_state}")
    if sensor_status != "within_10_percent":
        reasons.append(f"sensor_{sensor_status}")
    if not focal_present:
        reasons.append("missing_focal_length")
    return {
        "dimension_status": dimension_status,
        "sensor_status": sensor_status,
        "crop_state": crop_state,
        "strict_geometry_eligible": not reasons,
        "rejection_reasons": reasons,
    }


def _xmp_crop_state(path: Path) -> str:
    with path.open("rb") as stream:
        sample = stream.read(2 * 1024 * 1024).lower()
    if b"hascrop=\"true\"" in sample or b"hascrop>true<" in sample:
        return "true"
    if b"hascrop=\"false\"" in sample or b"hascrop>false<" in sample:
        return "false"
    return "missing"


def _read_exif(image: Image.Image) -> dict[str, Any]:
    raw = image.getexif()
    items = dict(raw.items())
    exif_ifd = getattr(getattr(ExifTags, "IFD", object()), "Exif", 34665)
    try:
        items.update(raw.get_ifd(exif_ifd))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return {
        ExifTags.TAGS.get(tag, str(tag)): value
        for tag, value in items.items()
    }


def audit(images_root: Path, sensor_table_path: Path) -> dict:
    sensor_table, table_hash = _load_sensor_table(sensor_table_path)
    paths = sorted(
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    counts = Counter()
    cameras = Counter()
    lenses = Counter()
    manifest_digest = hashlib.sha256()

    for path in paths:
        relative = path.relative_to(images_root).as_posix()
        content_hash = _sha256_file(path)
        manifest_digest.update(f"{relative}\t{content_hash}\n".encode())
        counts["images"] += 1
        try:
            with Image.open(path) as image:
                decoded_size = image.size
                exif = _read_exif(image)
        except (OSError, ValueError):
            counts["decode_error"] += 1
            continue

        if exif:
            counts["any_exif"] += 1
        if (_number(exif.get("FocalPlaneXResolution")) is not None
                and _number(exif.get("FocalPlaneYResolution")) is not None):
            counts["focal_plane_xy_resolution"] += 1
        for counter, key in (
            ("focal_length", "FocalLength"),
            ("focal_length_35mm", "FocalLengthIn35mmFilm"),
            ("aperture", "FNumber"),
            ("iso", "ISOSpeedRatings"),
            ("exposure_time", "ExposureTime"),
        ):
            if _number(exif.get(key)) is not None:
                counts[counter] += 1

        make = _text(exif.get("Make"))
        model = _text(exif.get("Model"))
        lens = _text(exif.get("LensModel"))
        if make or model:
            cameras[f"{make}|{model}"] += 1
        if lens:
            lenses[lens] += 1

        crop_state = _xmp_crop_state(path)
        classification = classify_geometry_record(
            decoded_size, exif, crop_state, sensor_table)
        dimension_status = classification["dimension_status"]
        sensor_status = classification["sensor_status"]
        counts[f"dimension_{dimension_status}"] += 1
        counts[f"sensor_{sensor_status}"] += 1
        counts[f"xmp_crop_{crop_state}"] += 1
        if _dimension_pair(exif) is not None:
            counts["exif_pixel_dimensions"] += 1
        if _derived_sensor(exif) is not None:
            counts["focal_plane_sensor_derived"] += 1
        if sensor_status in {"within_10_percent", "outlier"}:
            counts["sensor_table_model_match"] += 1
        if sensor_status == "within_10_percent":
            counts["sensor_diagonal_within_10_percent"] += 1
        elif sensor_status == "outlier":
            counts["sensor_diagonal_outlier"] += 1
        if classification["strict_geometry_eligible"]:
            counts["strict_geometry_eligible"] += 1
        else:
            primary = classification["rejection_reasons"][0]
            counts[f"strict_primary_rejection_{primary}"] += 1
            for reason in classification["rejection_reasons"]:
                counts[f"strict_rejection_{reason}"] += 1

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_tool_sha256": _sha256_file(Path(__file__).resolve()),
        "images_root_name": images_root.name,
        "source_manifest_content_sha256": manifest_digest.hexdigest(),
        "sensor_table_path": sensor_table_path.name,
        "sensor_table_sha256": table_hash,
        "counts": dict(sorted(counts.items())),
        "camera_distribution": dict(cameras.most_common()),
        "lens_distribution": dict(lenses.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Image directory to audit")
    parser.add_argument(
        "--sensor-table", type=Path,
        default=Path(__file__).parents[1] / "docs" / "camera_sensors_v1.json",
    )
    parser.add_argument("--output", type=Path, help="Write JSON here")
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    result = audit(args.images.resolve(), args.sensor_table.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
