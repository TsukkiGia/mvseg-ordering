from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


FAMILY_ROOTS: Dict[str, str] = {
    "experiment_acdc": "ACDC",
    "experiment_btcv": "BTCV",
    "experiment_buid": "BUID",
    "experiment_hipxray": "HipXRay",
    "experiment_pandental": "PanDental",
    "experiment_scd": "SCD",
    "experiment_scr": "SCR",
    "experiment_spineweb": "SpineWeb",
    "experiment_stare": "STARE",
    "experiment_t1mix": "T1mix",
    "experiment_wbc": "WBC",
    "experiment_total_segmentator": "TotalSegmentator",
}


def iter_family_task_dirs(
    repo_root: Path,
    *,
    procedure: str,
    include_families: Optional[Iterable[str]] = None,
) -> Iterator[Tuple[str, Path, str]]:
    """Yield (family, task_dir, root_name) for tasks under experiments/scripts[/procedure]."""
    scripts_root = repo_root / "experiments" / "scripts" / procedure

    allow: Optional[set[str]] = None
    if include_families:
        targets = set(include_families)
        allow = {
            root for root, fam in FAMILY_ROOTS.items() if fam in targets or root in targets
        }

    for root_name, family in FAMILY_ROOTS.items():
        if allow is not None and root_name not in allow:
            continue
        root_path = scripts_root / root_name
        if not root_path.exists():
            continue
        for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            yield family, task_dir, root_name
