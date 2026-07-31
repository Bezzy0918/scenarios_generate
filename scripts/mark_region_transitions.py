#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.patches import Polygon
from PIL import Image

from generate_scenario_from_llm import load_mapping
from mark_waypoints_by_region import Region, draw_region_overlays, load_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively mark required waypoint pairs for transitions between regions."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene directory containing navmesh_mask.png, topdown.json, topdown_mapping.json, and waypoints_by_region.json.",
    )
    parser.add_argument(
        "--waypoints",
        help="Optional explicit waypoints JSON path. Defaults to <scene-dir>/waypoints_by_region.json.",
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
        "--hide-waypoint-names",
        action="store_true",
        help="Draw waypoint markers without their waypoint ids.",
    )
    return parser.parse_args()


def load_waypoint_data(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if not isinstance(data.get("waypoints_by_region"), list):
        raise ValueError(f"{path} must contain waypoints_by_region")
    data.setdefault("region_transitions", [])
    return data


def flatten_waypoints(data: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for region in data.get("waypoints_by_region", []):
        if not isinstance(region, dict):
            continue
        region_id = region.get("region_id")
        label = region.get("label")
        for waypoint in region.get("waypoints", []):
            if not isinstance(waypoint, dict):
                continue
            waypoint_id = waypoint.get("id")
            pixel = waypoint.get("pixel")
            if not isinstance(waypoint_id, str) or not isinstance(pixel, list) or len(pixel) < 2:
                continue
            flat.append(
                {
                    "id": waypoint_id,
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "region_id": int(region_id),
                    "label": str(label),
                }
            )
    if not flat:
        raise ValueError("No valid waypoints found")
    return flat


def nearest_waypoint(
    waypoints: list[dict[str, Any]],
    pixel_x: float,
    pixel_y: float,
    max_distance: float,
) -> dict[str, Any] | None:
    best_waypoint: dict[str, Any] | None = None
    best_distance = float("inf")
    for waypoint in waypoints:
        point_x, point_y = waypoint["pixel"]
        distance = float(np.hypot(point_x - pixel_x, point_y - pixel_y))
        if distance < best_distance:
            best_distance = distance
            best_waypoint = waypoint
    if best_distance > max_distance:
        return None
    return best_waypoint


def transition_key(transition: dict[str, Any]) -> tuple[int, int, tuple[str, ...]]:
    return (
        int(transition["from_region_id"]),
        int(transition["to_region_id"]),
        tuple(transition["through_waypoints"]),
    )


def add_transition(data: dict[str, Any], first: dict[str, Any], second: dict[str, Any]) -> bool:
    transition = {
        "from_region_id": first["region_id"],
        "from_label": first["label"],
        "to_region_id": second["region_id"],
        "to_label": second["label"],
        "through_waypoints": [first["id"], second["id"]],
    }
    transitions = data.setdefault("region_transitions", [])
    existing = {transition_key(item) for item in transitions if isinstance(item, dict)}
    if transition_key(transition) in existing:
        return False
    transitions.append(transition)
    return True


def save_data(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(data.get('region_transitions', []))} transitions to {path}", flush=True)


def draw_existing_transitions(ax: plt.Axes, data: dict[str, Any], waypoint_by_id: dict[str, dict[str, Any]]) -> None:
    for transition in data.get("region_transitions", []):
        if not isinstance(transition, dict):
            continue
        waypoint_ids = transition.get("through_waypoints")
        if not isinstance(waypoint_ids, list) or len(waypoint_ids) != 2:
            continue
        first = waypoint_by_id.get(str(waypoint_ids[0]))
        second = waypoint_by_id.get(str(waypoint_ids[1]))
        if not first or not second:
            continue
        x1, y1 = first["pixel"]
        x2, y2 = second["pixel"]
        ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=2.4, alpha=0.9)
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "cyan", "lw": 2.0, "shrinkA": 6, "shrinkB": 6},
        )


def draw_waypoints(
    ax: plt.Axes,
    waypoints: list[dict[str, Any]],
    *,
    show_names: bool = True,
) -> None:
    for waypoint in waypoints:
        x, y = waypoint["pixel"]
        ax.scatter([x], [y], c="yellow", s=48, edgecolors="black", linewidths=0.8, zorder=5)
        if not show_names:
            continue
        ax.text(
            x + 5,
            y + 5,
            waypoint["id"],
            color="yellow",
            fontsize=7,
            weight="bold",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.2},
            zorder=6,
        )


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    waypoint_path = (
        Path(args.waypoints).expanduser().resolve()
        if args.waypoints
        else scene_dir / "waypoints_by_region.json"
    )
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

    mapping = load_mapping(mapping_path)
    regions: list[Region] = load_regions(regions_path)
    data = load_waypoint_data(waypoint_path)
    waypoints = flatten_waypoints(data)
    waypoint_by_id = {waypoint["id"]: waypoint for waypoint in waypoints}
    navmesh = Image.open(navmesh_path).convert("L")

    if navmesh.size != (mapping.image_width, mapping.image_height):
        raise ValueError(
            f"Navmesh size {navmesh.size} does not match mapping size "
            f"({mapping.image_width}, {mapping.image_height})"
        )

    selected: list[dict[str, Any]] = []
    selection_artists: list[Any] = []

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(navmesh, cmap="gray", vmin=0, vmax=255)
    draw_region_overlays(ax, regions)
    draw_existing_transitions(ax, data, waypoint_by_id)
    draw_waypoints(ax, waypoints, show_names=not args.hide_waypoint_names)
    ax.set_title("Click two waypoints for a transition; s save pair; backspace undo selection; q save and quit")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_xlim(-0.5, mapping.image_width - 0.5)
    ax.set_ylim(mapping.image_height - 0.5, -0.5)

    def redraw_selection() -> None:
        while selection_artists:
            selection_artists.pop().remove()
        for index, waypoint in enumerate(selected, start=1):
            x, y = waypoint["pixel"]
            selection_artists.append(
                ax.scatter([x], [y], c="red", s=120, edgecolors="white", linewidths=1.4, zorder=8)
            )
            selection_artists.append(
                ax.text(
                    x + 8,
                    y - 8,
                    f"{index}",
                    color="red",
                    fontsize=13,
                    weight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
                    zorder=9,
                )
            )
        fig.canvas.draw_idle()

    def select_waypoint(event: MouseEvent) -> None:
        if event.button != MouseButton.LEFT or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        waypoint = nearest_waypoint(waypoints, event.xdata, event.ydata, max_distance=18.0)
        if waypoint is None:
            print("No waypoint near click", flush=True)
            return
        if len(selected) == 2:
            selected.clear()
        if waypoint in selected:
            print(f"Already selected {waypoint['id']}", flush=True)
            return
        selected.append(waypoint)
        print(
            f"Selected {len(selected)}: {waypoint['id']} "
            f"region={waypoint['region_id']}:{waypoint['label']}",
            flush=True,
        )
        redraw_selection()

    def save_selected_transition() -> None:
        if len(selected) != 2:
            print("Select exactly two waypoints before saving a transition", flush=True)
            return
        first, second = selected
        if first["region_id"] == second["region_id"]:
            print("Warning: selected waypoints are in the same region", flush=True)
        added = add_transition(data, first, second)
        if added:
            print(f"Added transition {first['id']} -> {second['id']}", flush=True)
        else:
            print("Transition already exists", flush=True)
        save_data(waypoint_path, data)
        ax.clear()
        ax.imshow(navmesh, cmap="gray", vmin=0, vmax=255)
        draw_region_overlays(ax, regions)
        draw_existing_transitions(ax, data, waypoint_by_id)
        draw_waypoints(ax, waypoints, show_names=not args.hide_waypoint_names)
        ax.set_title("Click two waypoints for a transition; s save pair; backspace undo selection; q save and quit")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.set_xlim(-0.5, mapping.image_width - 0.5)
        ax.set_ylim(mapping.image_height - 0.5, -0.5)
        selected.clear()
        selection_artists.clear()
        fig.canvas.draw_idle()

    def on_key(event: KeyEvent) -> None:
        if event.key in {"backspace", "delete"}:
            if selected:
                removed = selected.pop()
                print(f"Unselected {removed['id']}", flush=True)
                redraw_selection()
        elif event.key == "s":
            save_selected_transition()
        elif event.key == "q":
            save_data(waypoint_path, data)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", select_waypoint)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Controls: left-click waypoint 1, left-click waypoint 2, s save transition, backspace undo selection, q save and quit")
    plt.show()


if __name__ == "__main__":
    main()
