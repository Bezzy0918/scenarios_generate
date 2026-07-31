#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from PIL import Image

from generate_scenario_from_llm import load_mapping


@dataclass(frozen=True)
class Region:
    region_id: int
    label: str
    shape_type: str
    points: list[list[float]]
    path: MplPath
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively mark candidate waypoints on a navmesh+region view, "
            "then save them grouped by topdown.json regions."
        )
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory containing navmesh_mask.png, topdown.json, and topdown_mapping.json.",
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
        "--mapping",
        help="Optional explicit topdown_mapping.json path. Defaults to <scene-dir>/topdown_mapping.json.",
    )
    parser.add_argument(
        "--output",
        help="Output waypoint JSON path. Defaults to <scene-dir>/waypoints_by_region.json.",
    )
    parser.add_argument(
        "--marker-color",
        default="red",
        help="Color used for clicked waypoint markers.",
    )
    parser.add_argument(
        "--hide-waypoint-names",
        action="store_true",
        help="Draw waypoint markers without their index and region labels.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Load the existing output JSON and append new waypoints without changing existing ids or transitions.",
    )
    return parser.parse_args()


def polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return abs(area) * 0.5


def load_regions(regions_path: Path) -> list[Region]:
    data = json.loads(regions_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{regions_path} does not contain a shapes list")

    regions: list[Region] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        points = shape.get("points")
        if shape.get("shape_type", "polygon") != "polygon" or not isinstance(points, list):
            continue
        polygon_points = [[float(point[0]), float(point[1])] for point in points]
        if len(polygon_points) < 3:
            continue
        label = str(shape.get("label") or f"region_{index}")
        regions.append(
            Region(
                region_id=index,
                label=label,
                shape_type=str(shape.get("shape_type", "polygon")),
                points=polygon_points,
                path=MplPath(np.array(polygon_points, dtype=np.float64)),
                area=polygon_area(polygon_points),
            )
        )
    if not regions:
        raise ValueError(f"{regions_path} does not contain valid polygon regions")
    return regions


def sanitize_id(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return sanitized or "region"


def find_region(regions: list[Region], pixel_x: int, pixel_y: int) -> Region | None:
    containing = [
        region
        for region in regions
        if region.path.contains_point((float(pixel_x), float(pixel_y)), radius=1e-6)
    ]
    if not containing:
        return None
    return min(containing, key=lambda region: region.area)


def waypoint_record(
    pixel_x: int,
    pixel_y: int,
    region: Region | None,
) -> dict[str, Any]:
    return {
        "pixel": [pixel_x, pixel_y],
        "region_id": region.region_id if region else None,
        "region_label": region.label if region else None,
    }


def build_output(
    regions: list[Region],
    waypoints: list[dict[str, Any]],
    region_transitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grouped: dict[int | None, list[dict[str, Any]]] = {region.region_id: [] for region in regions}
    grouped[None] = []

    region_counts: dict[int | None, int] = {region.region_id: 0 for region in regions}
    region_counts[None] = 0
    for waypoint in waypoints:
        waypoint_id = waypoint.get("id")
        if not isinstance(waypoint_id, str):
            continue
        match = re.search(r"_(\d+)$", waypoint_id)
        if match:
            region_id = waypoint["region_id"]
            region_counts[region_id] = max(region_counts[region_id], int(match.group(1)))

    for waypoint in waypoints:
        region_id = waypoint["region_id"]
        waypoint_id = waypoint.get("id")
        if not isinstance(waypoint_id, str):
            region_counts[region_id] += 1
            if region_id is None:
                waypoint_id = f"wp_unassigned_{region_counts[region_id]:03d}"
            else:
                label = sanitize_id(str(waypoint["region_label"]))
                waypoint_id = f"wp_r{region_id:02d}_{label}_{region_counts[region_id]:03d}"
        grouped[region_id].append({"id": waypoint_id, "pixel": waypoint["pixel"]})

    region_areas: list[dict[str, Any]] = []
    waypoints_by_region: list[dict[str, Any]] = []
    for region in regions:
        region_areas.append(
            {
                "region_id": region.region_id,
                "label": region.label,
                "polygon": region.points,
            }
        )
        waypoints_by_region.append(
            {
                "region_id": region.region_id,
                "label": region.label,
                "waypoints": grouped[region.region_id],
            }
        )

    output = {
        "region_areas": region_areas,
        "waypoints_by_region": waypoints_by_region,
    }
    if region_transitions is not None:
        output["region_transitions"] = region_transitions
    return output


def load_existing_waypoints(
    output_path: Path,
    regions: list[Region],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    groups = data.get("waypoints_by_region")
    if not isinstance(groups, list):
        raise ValueError(f"{output_path} does not contain waypoints_by_region")

    region_by_id = {region.region_id: region for region in regions}
    waypoints: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        region_id = group.get("region_id")
        if not isinstance(region_id, int) or region_id not in region_by_id:
            raise ValueError(f"Unknown region_id in {output_path}: {region_id}")
        region = region_by_id[region_id]
        for waypoint in group.get("waypoints", []):
            if not isinstance(waypoint, dict):
                continue
            waypoint_id = waypoint.get("id")
            pixel = waypoint.get("pixel")
            if not isinstance(waypoint_id, str) or not isinstance(pixel, list) or len(pixel) < 2:
                raise ValueError(f"Invalid waypoint in {output_path}: {waypoint}")
            waypoints.append(
                {
                    "id": waypoint_id,
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "region_id": region_id,
                    "region_label": region.label,
                }
            )

    transitions = data.get("region_transitions", [])
    if not isinstance(transitions, list):
        raise ValueError(f"region_transitions in {output_path} must be a list")
    return waypoints, [item for item in transitions if isinstance(item, dict)]


def save_waypoints(
    output_path: Path,
    regions: list[Region],
    waypoints: list[dict[str, Any]],
    region_transitions: list[dict[str, Any]] | None = None,
) -> None:
    output = build_output(
        regions=regions,
        waypoints=waypoints,
        region_transitions=region_transitions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(waypoints)} waypoints to {output_path}", flush=True)


def draw_region_overlays(
    ax: plt.Axes,
    regions: list[Region],
    *,
    show_ids: bool = False,
) -> None:
    color_map = plt.get_cmap("tab20")
    for index, region in enumerate(regions):
        color = color_map(index % color_map.N)
        ax.add_patch(
            Polygon(
                region.points,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                alpha=0.95,
            )
        )
        xs = [point[0] for point in region.points]
        ys = [point[1] for point in region.points]
        display_label = f"{region.label} ({region.region_id})" if show_ids else region.label
        ax.text(
            sum(xs) / len(xs),
            sum(ys) / len(ys),
            display_label,
            color=color,
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
        )


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
    mapping_path = (
        Path(args.mapping).expanduser().resolve()
        if args.mapping
        else scene_dir / "topdown_mapping.json"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else scene_dir / "waypoints_by_region.json"
    )

    mapping = load_mapping(mapping_path)
    regions = load_regions(regions_path)
    navmesh = Image.open(navmesh_path).convert("L")
    if navmesh.size != (mapping.image_width, mapping.image_height):
        raise ValueError(
            f"Navmesh size {navmesh.size} does not match mapping size "
            f"({mapping.image_width}, {mapping.image_height})"
        )

    if args.append:
        if not output_path.is_file():
            raise FileNotFoundError(f"Cannot append; waypoint JSON not found: {output_path}")
        waypoints, region_transitions = load_existing_waypoints(output_path, regions)
        print(f"Loaded {len(waypoints)} existing waypoints from {output_path}", flush=True)
    else:
        waypoints = []
        region_transitions = None
    initial_waypoint_count = len(waypoints)
    point_artists: list[Any] = []

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(navmesh, cmap="gray", vmin=0, vmax=255)
    draw_region_overlays(ax, regions)
    ax.set_title("Click to mark waypoints; right-click/backspace undo; s save; q save and quit")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_xlim(-0.5, mapping.image_width - 0.5)
    ax.set_ylim(mapping.image_height - 0.5, -0.5)

    def redraw_points() -> None:
        while point_artists:
            artist = point_artists.pop()
            artist.remove()
        for index, waypoint in enumerate(waypoints, start=1):
            pixel_x, pixel_y = waypoint["pixel"]
            label = waypoint["region_label"] or "unassigned"
            point_artists.append(
                ax.scatter([pixel_x], [pixel_y], c=args.marker_color, s=45, edgecolors="white", linewidths=0.8)
            )
            if not args.hide_waypoint_names:
                point_artists.append(
                    ax.text(
                        pixel_x + 6,
                        pixel_y + 6,
                        f"{index}: {label}",
                        color=args.marker_color,
                        fontsize=8,
                        weight="bold",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                    )
                )
        fig.canvas.draw_idle()

    def add_point(event: MouseEvent) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        pixel_x = max(0, min(mapping.image_width - 1, int(round(event.xdata))))
        pixel_y = max(0, min(mapping.image_height - 1, int(round(event.ydata))))
        region = find_region(regions, pixel_x, pixel_y)
        waypoint = waypoint_record(pixel_x, pixel_y, region)
        waypoints.append(waypoint)
        print(
            f"Added pixel=({pixel_x}, {pixel_y}) "
            f"region={waypoint['region_label'] or 'unassigned'}",
            flush=True,
        )
        redraw_points()
        save_waypoints(output_path, regions, waypoints, region_transitions)

    def undo_point() -> None:
        if not waypoints:
            return
        if args.append and len(waypoints) <= initial_waypoint_count:
            print("Append mode cannot remove existing waypoints; only newly added points can be undone", flush=True)
            return
        removed = waypoints.pop()
        print(
            f"Removed pixel=({removed['pixel'][0]}, {removed['pixel'][1]}) "
            f"region={removed['region_label'] or 'unassigned'}",
            flush=True,
        )
        redraw_points()
        save_waypoints(output_path, regions, waypoints, region_transitions)

    def on_click(event: MouseEvent) -> None:
        if event.button == MouseButton.LEFT:
            add_point(event)
        elif event.button == MouseButton.RIGHT:
            undo_point()

    def on_key(event: KeyEvent) -> None:
        if event.key in {"backspace", "delete"}:
            undo_point()
        elif event.key == "s":
            save_waypoints(output_path, regions, waypoints, region_transitions)
        elif event.key == "q":
            save_waypoints(output_path, regions, waypoints, region_transitions)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw_points()
    print("Controls: left-click add, right-click/backspace undo, s save, q save and quit", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
