#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REQUIRED_SCENE_FILES = (
    "overlay.png",
    "navmesh_mask.png",
    "topdown_mapping.json",
    "waypoints_by_region.json",
    "scene_prompt.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple distinct trajectory scenarios for one prepared scene."
    )
    parser.add_argument("--scene-dir", required=True, help="Prepared scene directory.")
    parser.add_argument("--output-root", required=True, help="Root directory for all generated outputs.")
    parser.add_argument("--llm-config", required=True, help="LLM YAML configuration path.")
    parser.add_argument("--prompt", required=True, help="Common scenario request for every variant.")
    parser.add_argument("--count", type=int, default=5, help="Number of distinct variants. Default: 5.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum generation attempts per variant. Default: 3.",
    )
    parser.add_argument(
        "--pedestrians",
        type=int,
        default=2,
        help="Required number of pedestrian agents. Default: 2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate variants whose output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned variant output paths without calling the LLM.",
    )
    return parser.parse_args()


def waypoint_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("id", "waypoint_id", "name"):
            if isinstance(value.get(key), str):
                return value[key]
    return str(value)


def route_from_agent(agent: dict[str, Any]) -> tuple[str, ...]:
    start = agent.get("spawn_waypoint") or agent.get("start_waypoint") or agent.get("spawn")
    route = [waypoint_id(start)] if start is not None else []
    raw_waypoints = agent.get("waypoints") or agent.get("waypoint_ids") or []
    if isinstance(raw_waypoints, list):
        route.extend(waypoint_id(value) for value in raw_waypoints)
    return tuple(route)


def route_from_robot(robot: dict[str, Any]) -> tuple[str, ...]:
    start = robot.get("start_waypoint") or robot.get("spawn_waypoint") or robot.get("start")
    route = [waypoint_id(start)] if start is not None else []
    raw_waypoints = robot.get("waypoints")
    if isinstance(raw_waypoints, list) and raw_waypoints:
        route.extend(waypoint_id(value) for value in raw_waypoints)
    else:
        goal = robot.get("goal_waypoint") or robot.get("goal")
        if goal is not None:
            route.append(waypoint_id(goal))
    return tuple(route)


def scenario_routes(data: Any, expected_pedestrians: int) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    agents = data.get("agents")
    if not isinstance(agents, list) or len(agents) != expected_pedestrians:
        actual = len(agents) if isinstance(agents, list) else 0
        raise ValueError(f"expected {expected_pedestrians} agents, got {actual}")
    if not all(isinstance(agent, dict) for agent in agents):
        raise ValueError("every agent must be a JSON object")
    robot = data.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("missing robot object")

    agent_routes = [route_from_agent(agent) for agent in agents]
    robot_route = route_from_robot(robot)
    if any(len(route) < 3 for route in agent_routes):
        raise ValueError("each pedestrian route must contain a spawn and at least two waypoints")
    if len(robot_route) < 2:
        raise ValueError("robot route must contain start and goal waypoints")
    return agent_routes, robot_route


def scenario_signature(
    agent_routes: list[tuple[str, ...]],
    robot_route: tuple[str, ...],
) -> str:
    # Sorting makes a simple hunav_1/hunav_2 name swap count as the same scenario.
    normalized = {"agents": sorted(agent_routes), "robot": robot_route}
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def route_summary(
    variant_number: int,
    agent_routes: list[tuple[str, ...]],
    robot_route: tuple[str, ...],
) -> str:
    parts = [
        f"候选 {variant_number} 行人{i}: {' -> '.join(route)}"
        for i, route in enumerate(agent_routes, start=1)
    ]
    parts.append(f"候选 {variant_number} 机器人: {' -> '.join(robot_route)}")
    return "\n".join(parts)


def variant_prompt(
    common_prompt: str,
    variant_number: int,
    count: int,
    pedestrians: int,
    previous_summaries: list[str],
    attempt: int,
) -> str:
    exclusion = ""
    if previous_summaries:
        exclusion = (
            "\n以下路线已经用于先前候选，本次不得生成完全相同的行人和机器人路线组合：\n"
            + "\n".join(previous_summaries)
        )
    retry_rule = ""
    if attempt > 1:
        retry_rule = f"\n这是该候选的第 {attempt} 次尝试，请明显调整起点、终点或主要路线。"
    return f"""{common_prompt}

这是同一场景的第 {variant_number}/{count} 个独立候选方案。
必须恰好生成 {pedestrians} 个行人和 1 个机器人。
优先通过不同的起点、目标区域和 waypoint 路线形成有意义的差异；角色、模型或速度变化不能作为唯一差异。
仍须严格遵守场景连通规则、region_transitions 和 navmesh 约束。{exclusion}{retry_rule}"""


def load_existing_variant(
    llm_output: Path,
    expected_pedestrians: int,
) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    data = json.loads(llm_output.read_text(encoding="utf-8"))
    return scenario_routes(data, expected_pedestrians)


def render_scenario_preview(
    renderer: Path,
    scene_dir: Path,
    scenario_path: Path,
    preview_path: Path,
) -> None:
    command = [
        sys.executable,
        str(renderer),
        "--mapping",
        str(scene_dir / "topdown_mapping.json"),
        "--view",
        "overlay",
        "--scenario",
        str(scenario_path),
        "--save-overlay",
        str(preview_path),
        "--no-show",
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to render preview for {scenario_path}")


def create_overview(preview_paths: list[Path], output_path: Path) -> None:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    thumbnails: list[Image.Image] = []
    for preview_path in preview_paths:
        with Image.open(preview_path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((800, 800), resampling)
            thumbnails.append(thumbnail)

    columns = min(2, len(thumbnails))
    rows = math.ceil(len(thumbnails) / columns)
    margin = 12
    label_height = 32
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails) + label_height
    overview = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(overview)
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        cell_x = margin + column * (cell_width + margin)
        cell_y = margin + row * (cell_height + margin)
        image_x = cell_x + (cell_width - thumbnail.width) // 2
        image_y = cell_y + label_height
        draw.text((cell_x + 6, cell_y + 8), f"scenario_{index + 1:03d}", fill="black")
        overview.paste(thumbnail, (image_x, image_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overview.save(output_path)


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.pedestrians < 1:
        raise ValueError("--pedestrians must be at least 1")

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    llm_config = Path(args.llm_config).expanduser().resolve()
    generator = Path(__file__).with_name("generate_scenario_from_llm.py").resolve()
    renderer = Path(__file__).with_name("topdown_click_to_world.py").resolve()

    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    missing = [name for name in REQUIRED_SCENE_FILES if not (scene_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Scene is not ready; missing: {', '.join(missing)}")
    if not llm_config.is_file():
        raise FileNotFoundError(f"LLM config not found: {llm_config}")

    scene_output_root = output_root / scene_dir.name
    signatures: set[str] = set()
    summaries: list[str] = []
    generated = 0
    existing = 0

    for variant_number in range(1, args.count + 1):
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        llm_output = variant_dir / "llm_agents.json"
        scenario_output = variant_dir / "generated_scenario.yaml"

        if not args.overwrite and llm_output.is_file() and scenario_output.is_file():
            try:
                agent_routes, robot_route = load_existing_variant(llm_output, args.pedestrians)
                signature = scenario_signature(agent_routes, robot_route)
                if signature in signatures:
                    print(f"DUPLICATE EXISTING scenario_{variant_number:03d}; use --overwrite")
                    raise SystemExit(1)
                signatures.add(signature)
                summaries.append(route_summary(variant_number, agent_routes, robot_route))
                existing += 1
                print(f"EXISTS scenario_{variant_number:03d}")
                continue
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"INVALID EXISTING scenario_{variant_number:03d}: {exc}; use --overwrite")
                raise SystemExit(1) from exc

        if args.dry_run:
            print(f"READY scenario_{variant_number:03d}: {variant_dir}")
            continue

        accepted = False
        for attempt in range(1, args.max_attempts + 1):
            prompt = variant_prompt(
                args.prompt,
                variant_number,
                args.count,
                args.pedestrians,
                summaries,
                attempt,
            )
            command = [
                sys.executable,
                str(generator),
                "--scene-dir",
                str(scene_dir),
                "--llm-config",
                str(llm_config),
                "--prompt",
                prompt,
                "--save-llm-output",
                str(llm_output),
                "--output",
                str(scenario_output),
            ]
            print(
                f"GENERATING scenario_{variant_number:03d} attempt {attempt}/{args.max_attempts}",
                flush=True,
            )
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                print(f"Attempt failed with exit code {result.returncode}", flush=True)
                continue
            try:
                agent_routes, robot_route = load_existing_variant(llm_output, args.pedestrians)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Rejected invalid output: {exc}", flush=True)
                continue

            signature = scenario_signature(agent_routes, robot_route)
            if signature in signatures:
                print("Rejected duplicate waypoint route combination", flush=True)
                continue

            signatures.add(signature)
            summaries.append(route_summary(variant_number, agent_routes, robot_route))
            generated += 1
            accepted = True
            print(f"GENERATED scenario_{variant_number:03d}", flush=True)
            break

        if not accepted:
            print(
                f"FAILED scenario_{variant_number:03d}: no valid distinct output after "
                f"{args.max_attempts} attempts",
                flush=True,
            )
            raise SystemExit(1)

    if args.dry_run:
        print(f"Summary: requested={args.count}, generated=0, existing={existing}")
        return

    preview_paths: list[Path] = []
    for variant_number in range(1, args.count + 1):
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        scenario_path = variant_dir / "generated_scenario.yaml"
        preview_path = variant_dir / "scenario_preview.png"
        print(f"RENDERING scenario_{variant_number:03d}", flush=True)
        render_scenario_preview(renderer, scene_dir, scenario_path, preview_path)
        preview_paths.append(preview_path)

    overview_path = scene_output_root / "scenarios_overview.png"
    create_overview(preview_paths, overview_path)
    print(f"WROTE OVERVIEW {overview_path}", flush=True)
    print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")


if __name__ == "__main__":
    main()
