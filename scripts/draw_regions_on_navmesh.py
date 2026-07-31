#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw topdown.json regions on navmesh_mask.png.")
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene output directory containing navmesh_mask.png and topdown.json.",
    )
    parser.add_argument(
        "--navmesh-mask",
        help="Optional explicit navmesh mask path. Defaults to <scene-dir>/navmesh_mask.png.",
    )
    parser.add_argument(
        "--regions",
        help="Optional explicit region JSON path. Defaults to <scene-dir>/topdown.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the rendered output image.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Draw polygon outlines without text labels.",
    )
    parser.add_argument(
        "--show-region-ids",
        action="store_true",
        help="Append each region id to its label, for example office (2).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving.",
    )
    return parser.parse_args()


def load_shapes(regions_path: Path) -> list[dict[str, Any]]:
    data = json.loads(regions_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{regions_path} does not contain a shapes list")
    return [
        shape
        for shape in shapes
        if isinstance(shape, dict)
        and shape.get("shape_type", "polygon") == "polygon"
        and isinstance(shape.get("points"), list)
    ]


def polygon_center(points: list[list[float]]) -> tuple[float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def draw_regions(
    navmesh_path: Path,
    regions_path: Path,
    output_path: Path,
    draw_labels: bool,
    show_region_ids: bool,
    show: bool,
) -> None:
    navmesh = Image.open(navmesh_path).convert("L")
    shapes = load_shapes(regions_path)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(navmesh, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Regions on navmesh mask")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    color_map = plt.get_cmap("tab20")
    for index, shape in enumerate(shapes):
        points = [[float(x), float(y)] for x, y in shape["points"]]
        color = color_map(index % color_map.N)
        patch = Polygon(
            points,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=2.2,
            alpha=0.95,
        )
        ax.add_patch(patch)

        if draw_labels:
            center_x, center_y = polygon_center(points)
            label = str(shape.get("label") or f"region_{index}")
            if show_region_ids:
                label = f"{label} ({index})"
            ax.text(
                center_x,
                center_y,
                label,
                color=color,
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {len(shapes)} regions to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    navmesh_path = (
        Path(args.navmesh_mask).expanduser().resolve()
        if args.navmesh_mask
        else scene_dir / "navmesh_mask.png"
    )
    regions_path = (
        Path(args.regions).expanduser().resolve()
        if args.regions
        else scene_dir / "topdown.json"
    )
    output_path = Path(args.output).expanduser().resolve()

    draw_regions(
        navmesh_path=navmesh_path,
        regions_path=regions_path,
        output_path=output_path,
        draw_labels=not args.no_labels,
        show_region_ids=args.show_region_ids,
        show=args.show,
    )


if __name__ == "__main__":
    main()
