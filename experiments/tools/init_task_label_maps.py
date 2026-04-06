#!/usr/bin/env python3
"""Initialize per-dataset task->label JSON files from task folder names."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experiments.dataset.task_id_parser import decode_task_component


TASK_DIR_PATTERN = re.compile(
    r"^(?P<task_component>.+)_label(?P<label>\d+)_(?P<slicing>midslice|maxslice)(?:_idx\d+)?$"
)


def _discover_task_keys(scripts_root: Path) -> dict[str, set[str]]:
    task_keys_by_dataset: dict[str, set[str]] = {}

    # Task directories are expected at:
    # - scripts/<procedure>/<experiment>/<task>
    # - scripts/<split>/<procedure>/<experiment>/<task>
    candidate_paths = list(scripts_root.glob("*/*/*")) + list(scripts_root.glob("*/*/*/*"))
    for path in candidate_paths:
        if not path.is_dir():
            continue
        match = TASK_DIR_PATTERN.match(path.name)
        if match is None:
            continue

        task_component = str(match.group("task_component"))
        label = int(match.group("label"))
        slicing = str(match.group("slicing"))
        mega_task = decode_task_component(task_component)
        family = mega_task.split("/", 1)[0]
        task_key = f"{mega_task}|label={label}|slicing={slicing}"
        task_keys_by_dataset.setdefault(family, set()).add(task_key)

    return task_keys_by_dataset


def _load_existing_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}, found {type(payload)}.")
    return {str(key): str(value) for key, value in payload.items()}


def build_maps(
    *,
    scripts_root: Path,
    out_root: Path,
    overwrite: bool,
) -> list[Path]:
    discovered = _discover_task_keys(scripts_root)
    written: list[Path] = []

    for dataset, task_keys in sorted(discovered.items()):
        dataset_dir = out_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_dir / "task_to_label.json"

        existing_map = {} if overwrite else _load_existing_map(out_path)
        for task_key in sorted(task_keys):
            existing_map.setdefault(task_key, "")

        ordered = {key: existing_map[key] for key in sorted(existing_map.keys())}
        out_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        written.append(out_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize per-dataset task label maps.")
    parser.add_argument(
        "--scripts-root",
        type=Path,
        default=Path("experiments/scripts"),
        help="Root directory to scan for task folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments/task_label_maps"),
        help="Output directory for per-dataset JSON map files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing map values instead of preserving manual edits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_maps(
        scripts_root=args.scripts_root,
        out_root=args.out_root,
        overwrite=bool(args.overwrite),
    )
    if not written:
        raise SystemExit(f"No task folders found under {args.scripts_root}.")
    print(f"Wrote {len(written)} dataset map files under {args.out_root}")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
