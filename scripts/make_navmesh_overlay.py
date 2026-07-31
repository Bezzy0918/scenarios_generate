#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a pixel-aligned topdown/navmesh overlay image.")
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory containing topdown.png and navmesh_mask.png.",
    )
    parser.add_argument(
        "--topdown",
        help="Optional explicit topdown image path. Defaults to <scene-dir>/topdown.png.",
    )
    parser.add_argument(
        "--navmesh-mask",
        help="Optional explicit navmesh mask path. Defaults to <scene-dir>/navmesh_mask.png.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output overlay PNG path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.38,
        help="Navmesh overlay alpha in [0, 1].",
    )
    parser.add_argument(
        "--green",
        default="66,135,92",
        help="Overlay RGB color as R,G,B.",
    )
    parser.add_argument(
        "--dim-non-walkable",
        type=float,
        default=0.18,
        help="Amount to dim non-walkable pixels in [0, 1]. Use 0 to disable.",
    )
    parser.add_argument(
        "--black-non-walkable",
        action="store_true",
        help="Render non-walkable pixels as black instead of dimmed topdown pixels.",
    )
    parser.add_argument(
        "--black-outside-region-bounds",
        action="store_true",
        help="Render pixels outside the bounding box of topdown.json regions as black.",
    )
    parser.add_argument(
        "--regions",
        help="Optional explicit topdown region JSON. Defaults to <scene-dir>/topdown.json.",
    )
    parser.add_argument(
        "--bounds-padding",
        type=int,
        default=0,
        help="Pixel padding added around the region bounding box.",
    )
    return parser.parse_args()


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError("--green must be formatted as R,G,B with values in 0..255")
    return parts[0], parts[1], parts[2]


def load_region_bounds(regions_path: Path, image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    data = json.loads(regions_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{regions_path} does not contain a shapes list")

    xs: list[float] = []
    ys: list[float] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        points = shape.get("points")
        if not isinstance(points, list):
            continue
        for point in points:
            if isinstance(point, list) and len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))

    if not xs or not ys:
        raise ValueError(f"{regions_path} does not contain polygon points")

    width, height = image_size
    left = max(0, math.floor(min(xs)) - padding)
    top = max(0, math.floor(min(ys)) - padding)
    right = min(width - 1, math.ceil(max(xs)) + padding)
    bottom = min(height - 1, math.ceil(max(ys)) + padding)
    return left, top, right, bottom


def inside_bounds(x: int, y: int, bounds: tuple[int, int, int, int] | None) -> bool:
    if bounds is None:
        return True
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom


def make_overlay(
    topdown_path: Path,
    navmesh_path: Path,
    output_path: Path,
    overlay_color: tuple[int, int, int],
    alpha: float,
    dim_non_walkable: float,
    black_non_walkable: bool,
    outside_black_bounds: tuple[int, int, int, int] | None,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if not 0.0 <= dim_non_walkable <= 1.0:
        raise ValueError("--dim-non-walkable must be in [0, 1]")

    topdown = Image.open(topdown_path).convert("RGBA")
    navmesh = Image.open(navmesh_path).convert("L")
    if navmesh.size != topdown.size:
        raise ValueError(f"Navmesh size {navmesh.size} does not match topdown size {topdown.size}")

    output = topdown.copy()
    pixels = output.load()
    mask_pixels = navmesh.load()

    overlay_alpha = int(round(alpha * 255))
    for y in range(output.height):
        for x in range(output.width):
            r, g, b, a = pixels[x, y]
            if not inside_bounds(x, y, outside_black_bounds):
                pixels[x, y] = (0, 0, 0, a)
            elif mask_pixels[x, y] >= 250:
                r = int(round(r * (1.0 - alpha) + overlay_color[0] * alpha))
                g = int(round(g * (1.0 - alpha) + overlay_color[1] * alpha))
                b = int(round(b * (1.0 - alpha) + overlay_color[2] * alpha))
                pixels[x, y] = (r, g, b, a)
            elif black_non_walkable:
                pixels[x, y] = (0, 0, 0, a)
            elif dim_non_walkable > 0:
                factor = 1.0 - dim_non_walkable
                pixels[x, y] = (
                    int(round(r * factor + 255 * dim_non_walkable)),
                    int(round(g * factor + 255 * dim_non_walkable)),
                    int(round(b * factor + 255 * dim_non_walkable)),
                    a,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.convert("RGB").save(output_path)
    print(f"Saved overlay: {output_path}")
    print(f"Size: {output.size[0]}x{output.size[1]}")


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    topdown_path = Path(args.topdown).expanduser().resolve() if args.topdown else scene_dir / "topdown.png"
    navmesh_path = (
        Path(args.navmesh_mask).expanduser().resolve()
        if args.navmesh_mask
        else scene_dir / "navmesh_mask.png"
    )
    output_path = Path(args.output).expanduser().resolve()
    regions_path = Path(args.regions).expanduser().resolve() if args.regions else scene_dir / "topdown.json"
    outside_black_bounds = None
    if args.black_outside_region_bounds:
        image_size = Image.open(topdown_path).size
        outside_black_bounds = load_region_bounds(regions_path, image_size, args.bounds_padding)

    make_overlay(
        topdown_path=topdown_path,
        navmesh_path=navmesh_path,
        output_path=output_path,
        overlay_color=parse_rgb(args.green),
        alpha=args.alpha,
        dim_non_walkable=args.dim_non_walkable,
        black_non_walkable=args.black_non_walkable,
        outside_black_bounds=outside_black_bounds,
    )


if __name__ == "__main__":
    main()
