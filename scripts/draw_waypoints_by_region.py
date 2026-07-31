#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from PIL import Image

from mark_region_transitions import (
    draw_existing_transitions,
    draw_waypoints,
    flatten_waypoints,
    load_waypoint_data,
)
from mark_waypoints_by_region import draw_region_overlays, load_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw all waypoint ids from waypoints_by_region.json on a scene image."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory containing waypoints_by_region.json and scene images.",
    )
    parser.add_argument(
        "--waypoints",
        help="Optional explicit waypoints JSON path. Defaults to <scene-dir>/waypoints_by_region.json.",
    )
    parser.add_argument(
        "--regions",
        help="Optional explicit region JSON path. Defaults to <scene-dir>/topdown.json.",
    )
    parser.add_argument(
        "--background",
        choices=["overlay", "topdown", "navmesh"],
        default="overlay",
        help="Background image to draw on. Default: overlay.",
    )
    parser.add_argument(
        "--output",
        help="Output image path. Defaults to <scene-dir>/waypoints_overview.png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Saved figure DPI. Default: 160.",
    )
    parser.add_argument(
        "--hide-regions",
        action="store_true",
        help="Do not draw region polygons and labels.",
    )
    parser.add_argument(
        "--show-region-ids",
        action="store_true",
        help="Append each region id to its label, for example office (2).",
    )
    parser.add_argument(
        "--hide-transitions",
        action="store_true",
        help="Do not draw region transition arrows.",
    )
    parser.add_argument(
        "--hide-waypoint-names",
        action="store_true",
        help="Draw waypoint markers without their waypoint ids.",
    )
    return parser.parse_args()


def background_path(scene_dir: Path, background: str) -> Path:
    candidates = {
        "overlay": [scene_dir / "overlay.png", scene_dir / "topdown.png", scene_dir / "navmesh_mask.png"],
        "topdown": [scene_dir / "topdown.png", scene_dir / "overlay.png", scene_dir / "navmesh_mask.png"],
        "navmesh": [scene_dir / "navmesh_mask.png", scene_dir / "overlay.png", scene_dir / "topdown.png"],
    }
    for path in candidates[background]:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No usable background image found in {scene_dir}; expected overlay.png, topdown.png, or navmesh_mask.png"
    )


def load_image(path: Path) -> Image.Image:
    if path.name == "navmesh_mask.png":
        return Image.open(path).convert("L")
    return Image.open(path).convert("RGB")


def draw_summary(ax: plt.Axes, waypoints: list[dict[str, Any]], waypoint_path: Path) -> None:
    region_counts: dict[str, int] = {}
    for waypoint in waypoints:
        label = str(waypoint["label"])
        region_counts[label] = region_counts.get(label, 0) + 1
    summary = ", ".join(f"{label}: {count}" for label, count in sorted(region_counts.items()))
    ax.set_title(f"{waypoint_path.name}: {len(waypoints)} waypoints ({summary})")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    waypoint_path = (
        Path(args.waypoints).expanduser().resolve()
        if args.waypoints
        else scene_dir / "waypoints_by_region.json"
    )
    regions_path = (
        Path(args.regions).expanduser().resolve()
        if args.regions
        else scene_dir / "topdown.json"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else scene_dir / "waypoints_overview.png"
    )

    data = load_waypoint_data(waypoint_path)
    waypoints = flatten_waypoints(data)
    waypoint_by_id = {waypoint["id"]: waypoint for waypoint in waypoints}
    image_path = background_path(scene_dir, args.background)
    image = load_image(image_path)

    fig_width = 12
    fig_height = max(8, fig_width * image.height / image.width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if image.mode == "L":
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(image)

    if not args.hide_regions:
        draw_region_overlays(ax, load_regions(regions_path), show_ids=args.show_region_ids)
    if not args.hide_transitions:
        draw_existing_transitions(ax, data, waypoint_by_id)
    draw_waypoints(ax, waypoints, show_names=not args.hide_waypoint_names)
    draw_summary(ax, waypoints, waypoint_path)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
