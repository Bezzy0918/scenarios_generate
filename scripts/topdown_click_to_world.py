#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.figure import Figure
from PIL import Image

@dataclass
class CameraRayMapping:
    image_width: int
    image_height: int
    camera_projection: np.ndarray
    camera_world_transform: np.ndarray
    ground_plane_z: float

    def world_to_pixel(self, world_x: float, world_y: float) -> tuple[float, float]:
        world_point = np.array([world_x, world_y, self.ground_plane_z, 1.0], dtype=np.float64)
        camera_point = world_point @ np.linalg.inv(self.camera_world_transform)
        clip_point = self.camera_projection @ camera_point
        if abs(clip_point[3]) < 1e-9:
            raise ValueError("Projected point is at infinity")
        ndc = clip_point[:3] / clip_point[3]
        pixel_x = ((ndc[0] + 1.0) * 0.5) * self.image_width - 0.5
        pixel_y = ((1.0 - ndc[1]) * 0.5) * self.image_height - 0.5
        return float(pixel_x), float(pixel_y)

    def pixel_to_world(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        ndc_x = 2.0 * ((pixel_x + 0.5) / self.image_width) - 1.0
        ndc_y = 1.0 - 2.0 * ((pixel_y + 0.5) / self.image_height)

        clip_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
        clip_far = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)

        inv_projection = np.linalg.inv(self.camera_projection)
        near_camera = inv_projection @ clip_near
        far_camera = inv_projection @ clip_far
        near_camera /= near_camera[3]
        far_camera /= far_camera[3]

        # Replicator/USD transform matrices here use row-vector convention:
        # homogeneous points transform as p_world = p_camera @ T_camera_to_world.
        near_world = near_camera @ self.camera_world_transform
        far_world = far_camera @ self.camera_world_transform
        near_world /= near_world[3]
        far_world /= far_world[3]

        ray_direction = far_world[:3] - near_world[:3]
        if abs(ray_direction[2]) < 1e-9:
            raise ValueError("Camera ray is parallel to the ground plane")

        t = (self.ground_plane_z - near_world[2]) / ray_direction[2]
        hit_world = near_world[:3] + t * ray_direction
        return float(hit_world[0]), float(hit_world[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click a top-down GRScenes image to inspect world coordinates.")
    parser.add_argument(
        "--image",
        help="Path to the top-down image. Defaults to <mapping-dir>/<image field from mapping json>.",
    )
    parser.add_argument(
        "--view",
        choices=("topdown", "navmesh", "overlay"),
        default="topdown",
        help="Image view to display: topdown, navmesh mask, or topdown with navmesh overlay.",
    )
    parser.add_argument(
        "--navmesh-mask",
        help="Path to navmesh_mask.png. Defaults to <mapping-dir>/navmesh_mask.png when needed.",
    )
    parser.add_argument(
        "--navmesh-alpha",
        type=float,
        default=0.35,
        help="Alpha used when --view overlay is selected.",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help="Path to topdown_mapping.json produced by extract_grscene_products.py.",
    )
    parser.add_argument(
        "--marker-color",
        default="red",
        help="Marker color for clicked points.",
    )
    parser.add_argument(
        "--scenario",
        help="Optional HuNav scenario YAML. Draw each agent pose and waypoints on the top-down image.",
    )
    parser.add_argument(
        "--save-overlay",
        help="Optional path to save the image with scenario waypoints overlaid.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save/check the overlay without opening the interactive matplotlib window.",
    )
    return parser.parse_args()


def load_mapping(mapping_path: Path) -> tuple[CameraRayMapping, Path]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    image_path = mapping_path.parent / data["image"]
    camera_geometry = data.get("camera_geometry")
    if not camera_geometry or camera_geometry.get("method") != "camera_ray_ground_intersection":
        raise ValueError("Mapping file does not contain camera_geometry for ray-based world conversion")

    mapping = CameraRayMapping(
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        camera_projection=np.array(camera_geometry["camera_projection"], dtype=np.float64).reshape((4, 4), order="F"),
        camera_world_transform=np.array(camera_geometry["camera_world_transform"], dtype=np.float64).reshape((4, 4)),
        ground_plane_z=float(camera_geometry["ground_plane_z"]),
    )
    return mapping, image_path


def load_navmesh_mask(navmesh_path: Path, expected_size: tuple[int, int]) -> Image.Image:
    if not navmesh_path.exists():
        raise FileNotFoundError(f"Missing navmesh mask: {navmesh_path}")
    navmesh = Image.open(navmesh_path).convert("L")
    if navmesh.size != expected_size:
        raise ValueError(f"Navmesh size {navmesh.size} does not match image size {expected_size}")
    return navmesh


def format_click(pixel_x: int, pixel_y: int, world_x: float, world_y: float) -> str:
    return f"pixel=({pixel_x}, {pixel_y}) world=({world_x:.4f}, {world_y:.4f})"


def load_scenario_routes(scenario_path: Path) -> list[tuple[dict, bool]]:
    data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("dynamic"), list):
        raise ValueError("Scenario YAML must contain a dynamic list")
    agents = data["dynamic"]
    if not all(isinstance(agent, dict) for agent in agents):
        raise ValueError("Every entry in scenario dynamic must be a mapping")
    routes = [(agent, False) for agent in agents]

    robot = data.get("robot")
    if robot is not None:
        if not isinstance(robot, dict):
            raise ValueError("Scenario robot entry must be a mapping")
        robot = dict(robot)
        robot["name"] = "robot"
        routes.append((robot, True))
    return routes


def draw_scenario_waypoints(ax: Axes, mapping: CameraRayMapping, scenario_path: Path) -> None:
    routes = load_scenario_routes(scenario_path)
    color_map = plt.get_cmap("tab10")

    for agent_index, (agent, is_robot) in enumerate(routes):
        name = str(agent.get("name") or f"agent_{agent_index + 1}")
        pose = agent.get("pose")
        waypoints = agent.get("waypoints") or []
        if not isinstance(pose, list) or len(pose) < 2:
            raise ValueError(f"{name} pose must be [x, y, yaw]")
        if not isinstance(waypoints, list):
            raise ValueError(f"{name} waypoints must be a list")

        route_world = [pose, *waypoints]
        route_pixels: list[tuple[float, float]] = []
        for point_index, point in enumerate(route_world):
            if not isinstance(point, list) or len(point) < 2:
                raise ValueError(f"{name} route point {point_index} must be [x, y, yaw]")
            route_pixels.append(mapping.world_to_pixel(float(point[0]), float(point[1])))

        color = "black" if is_robot else color_map(agent_index % color_map.N)
        marker = "D" if is_robot else "s"
        line_style = "--" if is_robot else "-"
        xs = [point[0] for point in route_pixels]
        ys = [point[1] for point in route_pixels]

        if len(route_pixels) > 1:
            ax.plot(xs, ys, color=color, linewidth=2.8 if is_robot else 2.2, linestyle=line_style, alpha=0.95)
            for start, end in zip(route_pixels, route_pixels[1:]):
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "lw": 1.8,
                        "shrinkA": 8,
                        "shrinkB": 8,
                    },
                )

        ax.scatter([xs[0]], [ys[0]], marker=marker, s=95 if is_robot else 70, color=color, edgecolors="white", linewidths=1.2)
        ax.text(
            xs[0] + 7,
            ys[0] - 7,
            f"{name}: S",
            color=color,
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )

        if len(route_pixels) > 1:
            waypoint_xs = xs[1:]
            waypoint_ys = ys[1:]
            ax.scatter(
                waypoint_xs,
                waypoint_ys,
                marker="D" if is_robot else "o",
                s=70 if is_robot else 55,
                color=color,
                edgecolors="white",
                linewidths=1.0,
            )
            for waypoint_index, (pixel_x, pixel_y) in enumerate(route_pixels[1:], start=1):
                ax.text(
                    pixel_x + 7,
                    pixel_y + 7,
                    str(waypoint_index),
                    color=color,
                    fontsize=9,
                    weight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
                )


def connect_click_handler(
    fig: Figure,
    ax: Axes,
    mapping: CameraRayMapping,
    marker_color: str,
) -> None:
    def on_click(event: MouseEvent) -> None:
        if event.button != MouseButton.LEFT or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        pixel_x = max(0, min(mapping.image_width - 1, int(round(event.xdata))))
        pixel_y = max(0, min(mapping.image_height - 1, int(round(event.ydata))))
        world_x, world_y = mapping.pixel_to_world(pixel_x, pixel_y)
        message = format_click(pixel_x, pixel_y, world_x, world_y)
        print(message, flush=True)

        ax.scatter([pixel_x], [pixel_y], c=marker_color, s=30)
        ax.text(pixel_x + 6, pixel_y + 6, f"({world_x:.2f}, {world_y:.2f})", color=marker_color, fontsize=9)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping).expanduser().resolve()
    mapping, default_image_path = load_mapping(mapping_path)
    image_path = Path(args.image).expanduser().resolve() if args.image else default_image_path.resolve()

    image = Image.open(image_path).convert("RGB")
    if image.size != (mapping.image_width, mapping.image_height):
        raise ValueError(
            f"Image size {image.size} does not match mapping size "
            f"{(mapping.image_width, mapping.image_height)}"
        )

    navmesh_path = (
        Path(args.navmesh_mask).expanduser().resolve()
        if args.navmesh_mask
        else mapping_path.parent / "navmesh_mask.png"
    )
    navmesh = None
    if args.view in {"navmesh", "overlay"}:
        navmesh = load_navmesh_mask(navmesh_path, image.size)

    fig, ax = plt.subplots(figsize=(12, 10))
    if args.view == "navmesh":
        ax.imshow(navmesh, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Navmesh mask: left click to inspect world coordinates")
    else:
        ax.imshow(image)
        if args.view == "overlay":
            ax.imshow(navmesh, cmap="Greens", vmin=0, vmax=255, alpha=args.navmesh_alpha)
            ax.set_title("Top-down with navmesh overlay: left click to inspect world coordinates")
        else:
            ax.set_title("Left click to inspect world coordinates")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    if args.scenario:
        scenario_path = Path(args.scenario).expanduser().resolve()
        draw_scenario_waypoints(ax, mapping, scenario_path)
        print(f"Loaded scenario: {scenario_path}", flush=True)
    connect_click_handler(fig, ax, mapping, args.marker_color)

    print(f"Loaded image: {image_path}", flush=True)
    if navmesh is not None:
        print(f"Loaded navmesh: {navmesh_path}", flush=True)
    print(f"Loaded mapping: {mapping_path}", flush=True)
    print("Left click a point to print its pixel/world coordinates. Close the window to exit.", flush=True)
    if args.save_overlay:
        save_overlay_path = Path(args.save_overlay).expanduser().resolve()
        save_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_overlay_path, dpi=180, bbox_inches="tight")
        print(f"Saved overlay: {save_overlay_path}", flush=True)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
