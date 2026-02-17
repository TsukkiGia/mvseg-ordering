#!/usr/bin/env python3
"""Summarize progress for a recipe YAML (done/running/pending by job and task).

This is a non-destructive status view over expected runs from a recipe config.
It expands MegaMedical entries the same way as `exp_yaml_launcher`, then checks
result markers and partial artifacts on disk.

Examples:
  python -m experiments.tools.recipe_progress \
    --config experiments/recipes/random/experiment_btcv.yaml

  python -m experiments.tools.recipe_progress \
    --config experiments/recipes/fixed_uncertainty/acdc_fixed_uncertainty.yaml \
    --only-plan B \
    --active-minutes 180 \
    --show-jobs 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


import pandas as pd
import yaml


PLAN_MARKERS = {
    "A": Path("A") / "results" / "support_images_summary.csv",
    "B": Path("B") / "subset_support_images_summary.csv",
}


@dataclass
class JobRecord:
    """Expected run target and observed status on disk."""

    experiment_name: str
    plan: str
    policy_name: str
    task_name: str
    script_dir: Path
    marker_path: Path
    status: str
    last_update_utc: str | None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _format_ts(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    return ts.isoformat(timespec="seconds")


def _resolve_existing_path(path_value: str | Path, repo_root: Path) -> Path:
    """Resolve potentially relative file path against repo root."""
    path = Path(path_value)
    if path.exists():
        return path
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return path


def _ordering_meta(path: Path) -> tuple[str, str]:
    """Return ordering (type, name) from config YAML."""
    if not path.exists():
        raise FileNotFoundError(f"ordering_config_path not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    ordering_type = str(cfg.get("type", "random")).strip().lower()
    ordering_name = str(cfg.get("name") or "").strip()
    if not ordering_type:
        raise ValueError(f"Invalid ordering config at {path}: missing non-empty 'type'")
    if not ordering_name:
        raise ValueError(f"Invalid ordering config at {path}: missing non-empty 'name'")
    return ordering_type, ordering_name


def _needs_auto_mega_expansion(merged_config: dict[str, Any]) -> bool:
    """True when recipe expects task expansion via MegaMedical task table."""
    if not merged_config.get("use_mega_dataset", False):
        return False
    if merged_config.get("mega_target_index") is not None:
        return False
    if merged_config.get("mega_task") is not None:
        return False
    if merged_config.get("mega_label") is not None:
        return False
    if merged_config.get("mega_slicing") is not None:
        return False
    return bool(merged_config.get("mega_dataset_name"))


def _expand_entry_from_existing_dirs(
    *,
    defaults: dict[str, Any],
    raw_experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Expand one recipe entry by scanning already-materialized task directories.

    This avoids loading MegaMedical task tables and is enough for run-progress
    monitoring once jobs have been queued/started.
    """
    merged_config = {**defaults, **raw_experiment}
    if not _needs_auto_mega_expansion(merged_config):
        return [raw_experiment]

    base_script_dir = Path(str(merged_config.get("script_dir", "")))
    if not base_script_dir:
        raise ValueError(f"Missing script_dir in experiment: {raw_experiment.get('name')}")

    base_root = base_script_dir.parent
    ablation_dir_name = base_script_dir.name
    materialized_ablation_dirs = sorted(
        p for p in base_root.glob(f"*/{ablation_dir_name}") if p.is_dir()
    )

    if not materialized_ablation_dirs:
        # No expanded task directories materialized yet. Keep one placeholder
        # entry so users still see the experiment as pending.
        return [raw_experiment]

    expanded: list[dict[str, Any]] = []
    base_name = str(raw_experiment.get("name", "exp"))
    for ablation_dir in materialized_ablation_dirs:
        task_dir = ablation_dir.parent
        entry = dict(raw_experiment)
        entry["script_dir"] = str(ablation_dir)
        entry["task_name"] = task_dir.name
        entry["name"] = f"{base_name}_{task_dir.name}"
        expanded.append(entry)
    return expanded


def _latest_artifact_time(script_dir: Path) -> datetime | None:
    """Return newest timestamp among run metadata and known partial artifacts."""
    if not script_dir.exists():
        return None

    candidates: list[Path] = []
    candidates.extend(script_dir.rglob("run_metadata.json"))
    candidates.extend(script_dir.rglob("append_log.jsonl"))
    candidates.extend(script_dir.rglob("support_images_summary.csv"))
    candidates.extend(script_dir.rglob("subset_support_images_summary.csv"))
    candidates.extend(script_dir.rglob("support_images_iterations.csv"))
    candidates.extend(script_dir.rglob("diffs.csv"))

    newest: datetime | None = None
    for artifact in candidates:
        try:
            artifact_time = datetime.fromtimestamp(artifact.stat().st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            continue
        if newest is None or artifact_time > newest:
            newest = artifact_time
    return newest


def _status_from_paths(marker_path: Path, script_dir: Path, *, active_minutes: int) -> tuple[str, str | None]:
    """
    Classify status:
      - done: final marker exists
      - running: partial artifacts exist and are recent
      - stalled: partial artifacts exist but old
      - pending: no artifacts yet
    """
    if marker_path.exists():
        marker_ts = datetime.fromtimestamp(marker_path.stat().st_mtime, tz=timezone.utc)
        return "done", _format_ts(marker_ts)

    latest = _latest_artifact_time(script_dir)
    if latest is None:
        return "pending", None

    active_cutoff = _now_utc() - timedelta(minutes=active_minutes)
    if latest >= active_cutoff:
        return "running", _format_ts(latest)
    return "stalled", _format_ts(latest)


def _build_expected_jobs(
    *,
    config_path: Path,
    only_plan: str | None,
    active_minutes: int,
) -> list[JobRecord]:
    repo_root = Path(__file__).resolve().parents[2]
    config_data = yaml.safe_load(config_path.read_text()) or {}
    defaults: dict[str, Any] = config_data.get("defaults", {})
    experiments: list[dict[str, Any]] = config_data.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in config: {config_path}")

    job_records: list[JobRecord] = []

    for raw_experiment in experiments:
        expanded_entries = _expand_entry_from_existing_dirs(
            defaults=defaults,
            raw_experiment=raw_experiment,
        )

        for expanded_experiment in expanded_entries:
            merged_cfg = {**defaults, **expanded_experiment}
            plans = list(expanded_experiment.get("plan", ["A"]))
            if only_plan:
                plans = [plan for plan in plans if plan == only_plan]
                if not plans:
                    continue

            base_script_dir = Path(str(merged_cfg.get("script_dir", "")))
            if not base_script_dir:
                raise ValueError(f"Missing script_dir in experiment: {expanded_experiment.get('name')}")

            policy_entries = expanded_experiment.get("policies") or [None]

            for plan in plans:
                if plan not in PLAN_MARKERS:
                    raise ValueError(f"Unsupported plan '{plan}' in config '{config_path}'")

                for policy_entry in policy_entries:
                    if policy_entry is None:
                        policy_name = "__no_policy__"
                        script_dir = base_script_dir
                    else:
                        ordering_cfg_path = policy_entry.get("ordering_config_path")
                        if not ordering_cfg_path:
                            raise ValueError("Each policy entry must include ordering_config_path")
                        resolved_ordering_path = _resolve_existing_path(ordering_cfg_path, repo_root)
                        _, policy_name = _ordering_meta(resolved_ordering_path)
                        script_dir = base_script_dir / policy_name

                    marker_path = script_dir / PLAN_MARKERS[plan]
                    status, last_update_utc = _status_from_paths(
                        marker_path,
                        script_dir,
                        active_minutes=active_minutes,
                    )

                    task_name = str(merged_cfg.get("task_name") or base_script_dir.parent.name)
                    experiment_name = str(expanded_experiment.get("name", "exp"))
                    job_records.append(
                        JobRecord(
                            experiment_name=experiment_name,
                            plan=plan,
                            policy_name=policy_name,
                            task_name=task_name,
                            script_dir=script_dir,
                            marker_path=marker_path,
                            status=status,
                            last_update_utc=last_update_utc,
                        )
                    )
    return job_records


def _print_summary(job_df: pd.DataFrame) -> None:
    total_jobs = int(len(job_df))
    status_counts = job_df["status"].value_counts().to_dict()
    done_jobs = int(status_counts.get("done", 0))
    running_jobs = int(status_counts.get("running", 0))
    stalled_jobs = int(status_counts.get("stalled", 0))
    pending_jobs = int(status_counts.get("pending", 0))
    done_pct = (100.0 * done_jobs / total_jobs) if total_jobs else 0.0

    print("== Job Progress ==")
    print(f"Total jobs: {total_jobs}")
    print(
        f"done={done_jobs} running={running_jobs} stalled={stalled_jobs} "
        f"pending={pending_jobs} done_pct={done_pct:.1f}%"
    )

    # One row per task with "how much left".
    task_summary = (
        job_df.groupby(["task_name"], as_index=False)
        .agg(
            total_jobs=("status", "size"),
            done_jobs=("status", lambda s: int((s == "done").sum())),
            running_jobs=("status", lambda s: int((s == "running").sum())),
            stalled_jobs=("status", lambda s: int((s == "stalled").sum())),
            pending_jobs=("status", lambda s: int((s == "pending").sum())),
        )
        .sort_values(["done_jobs", "running_jobs", "pending_jobs"], ascending=[False, False, False])
    )
    task_summary["left_jobs"] = task_summary["total_jobs"] - task_summary["done_jobs"]
    print("\n== Task Progress ==")
    print(task_summary.to_string(index=False))

    experiment_summary = (
        job_df.groupby(["experiment_name", "plan"], as_index=False)
        .agg(
            total_jobs=("status", "size"),
            done_jobs=("status", lambda s: int((s == "done").sum())),
            running_jobs=("status", lambda s: int((s == "running").sum())),
            stalled_jobs=("status", lambda s: int((s == "stalled").sum())),
            pending_jobs=("status", lambda s: int((s == "pending").sum())),
        )
        .sort_values(["experiment_name", "plan"])
    )
    print("\n== Experiment Progress ==")
    print(experiment_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show done/running/pending progress for a recipe YAML.")
    parser.add_argument("--config", type=Path, required=True, help="Recipe YAML path.")
    parser.add_argument("--only-plan", choices=["A", "B"], default=None, help="Restrict to one plan.")
    parser.add_argument(
        "--active-minutes",
        type=int,
        default=120,
        help="Recent artifact window for classifying a job as running (default: 120).",
    )
    parser.add_argument(
        "--show-jobs",
        type=int,
        default=0,
        help="If > 0, print up to N non-done jobs with status and paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    job_records = _build_expected_jobs(
        config_path=args.config,
        only_plan=args.only_plan,
        active_minutes=args.active_minutes,
    )
    if not job_records:
        print("No expected jobs found from recipe.")
        return

    job_df = pd.DataFrame([record.__dict__ for record in job_records])
    _print_summary(job_df)

    if args.show_jobs > 0:
        print("\n== Non-done Jobs ==")
        not_done = job_df[job_df["status"] != "done"].copy()
        not_done = not_done.sort_values(["status", "experiment_name", "task_name", "policy_name"])
        cols = [
            "status",
            "experiment_name",
            "plan",
            "task_name",
            "policy_name",
            "last_update_utc",
            "script_dir",
            "marker_path",
        ]
        print(not_done[cols].head(args.show_jobs).to_string(index=False))


if __name__ == "__main__":
    main()
