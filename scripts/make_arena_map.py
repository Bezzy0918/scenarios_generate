#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a simple Arena-compatible map folder from a GRScenes output directory."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory containing topdown_mapping.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output map directory. Defaults to <scene-dir>/map.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Map resolution in meters per pixel. Default: 0.05.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="World-meter padding added around scene bounds. Default: 1.0.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=300,
        help="Minimum map image side length in pixels. Default: 300.",
    )
    parser.add_argument(
        "--no-square",
        action="store_true",
        help="Do not force map.png to be square.",
    )
    parser.add_argument(
        "--occupied-thresh",
        type=float,
        default=0.65,
        help="occupied_thresh written to map.yaml. Default: 0.65.",
    )
    parser.add_argument(
        "--free-thresh",
        type=float,
        default=0.196,
        help="free_thresh written to map.yaml. Default: 0.196.",
    )
    parser.add_argument(
        "--type",
        dest="map_type",
        help="Optional type field written to map.yaml, for example indoor.",
    )
    return parser.parse_args()


def load_mapping(mapping_path: Path) -> dict[str, Any]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    for key in ("bounds_min", "bounds_max"):
        if not isinstance(data.get(key), list) or len(data[key]) < 2:
            raise ValueError(f"{mapping_path} is missing {key}")
    return data


def compute_map_geometry(
    mapping: dict[str, Any],
    resolution: float,
    padding: float,
    min_size: int,
    square: bool,
) -> tuple[int, int, tuple[float, float]]:
    min_x = float(mapping["bounds_min"][0]) - padding
    min_y = float(mapping["bounds_min"][1]) - padding
    max_x = float(mapping["bounds_max"][0]) + padding
    max_y = float(mapping["bounds_max"][1]) + padding

    width = max(min_size, math.ceil((max_x - min_x) / resolution))
    height = max(min_size, math.ceil((max_y - min_y) / resolution))
    if square:
        side = max(width, height)
        width = side
        height = side

    return width, height, (min_x, min_y)


def write_map_yaml(
    output_path: Path,
    resolution: float,
    origin: tuple[float, float],
    occupied_thresh: float,
    free_thresh: float,
    map_type: str | None,
) -> None:
    lines = [
        "image: map.png",
        f"resolution: {resolution}",
        f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.0]",
        f"occupied_thresh: {occupied_thresh}",
        f"free_thresh: {free_thresh}",
        "negate: 0",
    ]
    if map_type:
        lines.append(f"type: {map_type}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    mapping_path = scene_dir / "topdown_mapping.json"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else scene_dir / "map"
    )

    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.padding < 0:
        raise ValueError("--padding must be non-negative")
    if args.min_size <= 0:
        raise ValueError("--min-size must be positive")

    mapping = load_mapping(mapping_path)
    width, height, origin = compute_map_geometry(
        mapping=mapping,
        resolution=args.resolution,
        padding=args.padding,
        min_size=args.min_size,
        square=not args.no_square,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (width, height), color=255).save(output_dir / "map.png")
    write_map_yaml(
        output_path=output_dir / "map.yaml",
        resolution=args.resolution,
        origin=origin,
        occupied_thresh=args.occupied_thresh,
        free_thresh=args.free_thresh,
        map_type=args.map_type,
    )

    print(f"Wrote {output_dir / 'map.png'} ({width}x{height})")
    print(f"Wrote {output_dir / 'map.yaml'} origin=({origin[0]:.6f}, {origin[1]:.6f})")


if __name__ == "__main__":
    main()
