#!/usr/bin/env python3
"""Simulate start-nominated uncertainty policies using existing Plan B CSVs.

Primary endpoint:
  nominated-start uncertainty vs random subset means (paired by task/subset)

Secondary diagnostic:
  nominated-start uncertainty vs uncertainty subset means
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.analysis.planb_utils import load_planb_summaries
from experiments.analysis.start_selector_registry import get_selector, list_selectors
from experiments.dataset.task_id_parser import ensure_task_triple_columns

SUBSET_KEYS = ["family", "task_id", "subset_index"]


def _load_encoder_cfg(path: Path, key: str | None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"encoder config path not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if key is None:
        if not isinstance(payload, dict):
            raise ValueError(f"encoder config at {path} must be a mapping.")
        return dict(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"encoder config at {path} must be a mapping to use --encoder-config-key.")
    if key not in payload:
        raise ValueError(f"Key '{key}' not found in encoder config file: {path}")
    selected = payload[key]
    if not isinstance(selected, dict):
        raise ValueError(
            f"encoder config key '{key}' in {path} must map to a dictionary, got {type(selected)}."
        )
    return dict(selected)


def _encoder_tag(encoder_cfg: dict[str, Any], cfg_key: str | None) -> str:
    cfg_blob = json.dumps(encoder_cfg, sort_keys=True, default=str)
    digest = hashlib.sha1(cfg_blob.encode("utf-8")).hexdigest()[:12]
    type_tag = str(encoder_cfg.get("type", "unknown"))
    key_tag = str(cfg_key) if cfg_key else "root"
    return f"{type_tag}:{key_tag}:{digest}"


def _to_bool_array(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float).fillna(0.0).to_numpy(dtype=float) > 0.0
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).to_numpy(dtype=bool)


def summarize_permutations(
    df: pd.DataFrame,
    *,
    policy_col: str = "policy_name",
) -> pd.DataFrame:
    required = {
        "family",
        "task_id",
        "subset_index",
        policy_col,
        "permutation_index",
        "image_index",
        "image_id",
        "final_dice",
        "iterations_used",
        "prompt_limit",
        "mega_task",
        "mega_label",
        "mega_slicing",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {sorted(missing)}")

    meta_cols = [
        "family",
        "task_id",
        policy_col,
        "subset_index",
        "permutation_index",
    ]
    perm_rows: list[dict[str, object]] = []
    for keys, grp in df.groupby(meta_cols, sort=False):
        family, task_id, policy_name, subset_index, permutation_index = keys
        start_rows = grp[grp["image_index"].astype(int) == 0]
        if start_rows.empty:
            raise ValueError(
                "Could not derive start image: no image_index==0 row for "
                f"(family={family}, task_id={task_id}, subset={subset_index}, "
                f"policy={policy_name}, perm={permutation_index})."
            )
        start_ids = sorted(start_rows["image_id"].dropna().astype(int).unique().tolist())
        if len(start_ids) != 1:
            raise ValueError(
                "Expected exactly one start image_id for "
                f"(family={family}, task_id={task_id}, subset={subset_index}, "
                f"policy={policy_name}, perm={permutation_index}), got {start_ids}."
            )
        start_image_id = int(start_ids[0])
        prompt_limit = int(pd.to_numeric(grp["prompt_limit"], errors="coerce").dropna().max())
        iterations_arr = pd.to_numeric(grp["iterations_used"], errors="raise").to_numpy(dtype=float)
        final_arr = pd.to_numeric(grp["final_dice"], errors="raise").to_numpy(dtype=float)
        task_name = (
            str(grp["task_name"].iloc[0])
            if "task_name" in grp.columns and pd.notna(grp["task_name"].iloc[0])
            else str(task_id).split("/", 1)[-1]
        )
        perm_rows.append(
            {
                "family": str(family),
                "task_id": str(task_id),
                "task_name": task_name,
                "subset_index": int(subset_index),
                policy_col: str(policy_name),
                "permutation_index": int(permutation_index),
                "start_image_id": int(start_image_id),
                "mega_task": str(grp["mega_task"].iloc[0]),
                "mega_label": int(grp["mega_label"].iloc[0]),
                "mega_slicing": str(grp["mega_slicing"].iloc[0]),
                "prompt_limit": int(prompt_limit),
                "final_dice": float(np.mean(final_arr)),
                "iterations_used": float(np.mean(iterations_arr)),
            }
        )
    return pd.DataFrame(perm_rows)


def build_start_conditioned_tables(
    perm_df: pd.DataFrame,
    *,
    policy_col: str = "policy_name",
) -> pd.DataFrame:
    start_keys = SUBSET_KEYS + [policy_col, "start_image_id"]
    return (
        perm_df.groupby(start_keys, as_index=False)
        .agg(
            task_name=("task_name", "first"),
            mega_task=("mega_task", "first"),
            mega_label=("mega_label", "first"),
            mega_slicing=("mega_slicing", "first"),
            prompt_limit=("prompt_limit", "max"),
            final_dice=("final_dice", "mean"),
            iterations_used=("iterations_used", "mean"),
            n_permutations=("permutation_index", "nunique"),
        )
        .reset_index(drop=True)
    )


def build_subset_mean_tables(
    perm_df: pd.DataFrame,
    *,
    policy_col: str = "policy_name",
) -> pd.DataFrame:
    subset_keys = SUBSET_KEYS + [policy_col]
    return (
        perm_df.groupby(subset_keys, as_index=False)
        .agg(
            task_name=("task_name", "first"),
            mega_task=("mega_task", "first"),
            mega_label=("mega_label", "first"),
            mega_slicing=("mega_slicing", "first"),
            prompt_limit=("prompt_limit", "max"),
            final_dice=("final_dice", "mean"),
            iterations_used=("iterations_used", "mean"),
            n_permutations=("permutation_index", "nunique"),
        )
        .reset_index(drop=True)
    )


def build_subset_candidate_table(uncertainty_df: pd.DataFrame) -> pd.DataFrame:
    candidate_rows: list[dict[str, object]] = []
    meta_cols = SUBSET_KEYS
    for keys, grp in uncertainty_df.groupby(meta_cols, sort=False):
        family, task_id, subset_index = keys
        image_ids = sorted(grp["image_id"].dropna().astype(int).unique().tolist())
        if not image_ids:
            continue
        task_name = (
            str(grp["task_name"].iloc[0])
            if "task_name" in grp.columns and pd.notna(grp["task_name"].iloc[0])
            else str(task_id).split("/", 1)[-1]
        )
        candidate_rows.append(
            {
                "family": str(family),
                "task_id": str(task_id),
                "task_name": task_name,
                "subset_index": int(subset_index),
                "mega_task": str(grp["mega_task"].iloc[0]),
                "mega_label": int(grp["mega_label"].iloc[0]),
                "mega_slicing": str(grp["mega_slicing"].iloc[0]),
                "image_ids": image_ids,
                "prompt_limit": int(pd.to_numeric(grp["prompt_limit"], errors="coerce").dropna().max()),
            }
        )
    return pd.DataFrame(candidate_rows)


def load_embedding_cache(path: Path) -> dict[tuple[str, str, str, str, int, int], np.ndarray]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding cache at {path} must be a dictionary.")
    cache: dict[tuple[str, str, str, str, int, int], np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(key, tuple) or len(key) != 6:
            continue
        cache[tuple(key)] = np.asarray(value, dtype=np.float32)
    return cache


def save_embedding_cache(path: Path, cache: dict[tuple[str, str, str, str, int, int], np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(cache, fh)


def build_subset_embedding_map(
    subset_candidates: pd.DataFrame,
    *,
    encoder: torch.nn.Module,
    encoder_tag: str,
    data_split: str,
    device: torch.device,
    dataset_seed: int,
    embedding_cache: dict[tuple[str, str, str, str, int, int], np.ndarray],
) -> tuple[dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    from experiments.dataset.mega_medical_dataset import MegaMedicalDataset

    dataset_cache: dict[tuple[str, int, str], MegaMedicalDataset] = {}
    subset_embedding_map: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    stats = {"cache_hits": 0, "cache_misses": 0}

    encoder = encoder.to(device).eval()
    for row in subset_candidates.itertuples(index=False):
        subset_key = (str(row.family), str(row.task_id), int(row.subset_index))
        image_ids = np.asarray(list(row.image_ids), dtype=int)
        vectors: list[np.ndarray] = []
        ds_key = (str(row.mega_task), int(row.mega_label), str(row.mega_slicing))
        if ds_key not in dataset_cache:
            dataset_cache[ds_key] = MegaMedicalDataset(
                task=str(row.mega_task),
                label=int(row.mega_label),
                slicing=str(row.mega_slicing),
                split=str(data_split),
                seed=int(dataset_seed),
            )
        dataset = dataset_cache[ds_key]

        for image_id in image_ids:
            cache_key = (
                str(encoder_tag),
                str(data_split),
                str(row.family),
                str(row.task_id),
                int(row.subset_index),
                int(image_id),
            )
            if cache_key in embedding_cache:
                vec = np.asarray(embedding_cache[cache_key], dtype=np.float32).reshape(-1)
                stats["cache_hits"] += 1
            else:
                try:
                    image, _ = dataset.get_item_by_data_index(int(image_id))
                except Exception as exc:  # pragma: no cover - dataset-env specific
                    raise ValueError(
                        "Failed to fetch image for embedding "
                        f"(family={row.family}, task_id={row.task_id}, subset={row.subset_index}, "
                        f"image_id={image_id}, split={data_split}, task={row.mega_task}, "
                        f"label={row.mega_label}, slicing={row.mega_slicing})."
                    ) from exc
                image = image.to(device)
                with torch.no_grad():
                    emb = encoder(image)
                vec = emb.detach().cpu().to(torch.float32).reshape(-1).numpy()
                embedding_cache[cache_key] = vec
                stats["cache_misses"] += 1
            vectors.append(vec)
        emb_mat = np.vstack(vectors).astype(np.float32)
        subset_embedding_map[subset_key] = (image_ids, emb_mat)
    return subset_embedding_map, stats


def nominate_starts(
    subset_embedding_map: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    *,
    selectors: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_key, (image_ids, embeddings) in subset_embedding_map.items():
        family, task_id, subset_index = subset_key
        n_candidates = int(image_ids.size)
        for selector_name in selectors:
            selector = get_selector(selector_name)
            start_image_id = int(selector(image_ids, embeddings))
            rows.append(
                {
                    "family": str(family),
                    "task_id": str(task_id),
                    "subset_index": int(subset_index),
                    "selector": str(selector_name),
                    "start_image_id": int(start_image_id),
                    "n_candidates": n_candidates,
                }
            )
    return pd.DataFrame(rows)


def build_primary_subset_table(
    nominated_starts: pd.DataFrame,
    uncertainty_start_outcomes: pd.DataFrame,
    random_subset_outcomes: pd.DataFrame,
    *,
    uncertainty_policy_names: list[str],
    random_policy_name: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for policy_name in uncertainty_policy_names:
        uncertainty_policy = uncertainty_start_outcomes[
            uncertainty_start_outcomes["policy_name"] == str(policy_name)
        ].copy()
        uncertainty_policy = uncertainty_policy.rename(
            columns={
                "policy_name": "uncertainty_policy",
                "final_dice": "nominated_final_dice",
                "iterations_used": "nominated_iterations_used",
                "n_permutations": "nominated_n_permutations",
                "prompt_limit": "nominated_prompt_limit",
                "task_name": "task_name",
            }
        )
        policy_nominated = nominated_starts.copy()
        policy_nominated["uncertainty_policy"] = str(policy_name)
        merged = policy_nominated.merge(
            uncertainty_policy,
            on=SUBSET_KEYS + ["uncertainty_policy", "start_image_id"],
            how="left",
            validate="many_to_one",
        )
        merged = merged.merge(
            random_subset_outcomes.rename(
                columns={
                    "policy_name": "random_policy",
                    "final_dice": "random_final_dice",
                    "iterations_used": "random_iterations_used",
                    "n_permutations": "random_n_permutations",
                    "prompt_limit": "random_prompt_limit",
                }
            ),
            on=SUBSET_KEYS,
            how="left",
            validate="many_to_one",
        )
        merged["random_policy"] = str(random_policy_name)
        has_uncert = merged["nominated_final_dice"].notna()
        has_random = merged["random_final_dice"].notna()
        merged["matched_vs_random"] = has_uncert & has_random
        merged["coverage_status"] = np.select(
            [has_uncert & has_random, ~has_uncert & has_random, has_uncert & ~has_random],
            ["matched", "missing_uncertainty_start", "missing_random_baseline"],
            default="missing_both",
        )
        merged["delta_final_dice"] = merged["nominated_final_dice"] - merged["random_final_dice"]
        merged["delta_iterations_used_improvement"] = (
            merged["random_iterations_used"] - merged["nominated_iterations_used"]
        )
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_secondary_subset_table(
    nominated_starts: pd.DataFrame,
    uncertainty_start_outcomes: pd.DataFrame,
    uncertainty_subset_outcomes: pd.DataFrame,
    *,
    uncertainty_policy_names: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for policy_name in uncertainty_policy_names:
        uncertainty_policy = uncertainty_start_outcomes[
            uncertainty_start_outcomes["policy_name"] == str(policy_name)
        ].copy()
        uncertainty_policy = uncertainty_policy.rename(
            columns={
                "policy_name": "uncertainty_policy",
                "final_dice": "nominated_final_dice",
                "iterations_used": "nominated_iterations_used",
                "n_permutations": "nominated_n_permutations",
                "prompt_limit": "nominated_prompt_limit",
            }
        )
        subset_mean = uncertainty_subset_outcomes[
            uncertainty_subset_outcomes["policy_name"] == str(policy_name)
        ].copy()
        subset_mean = subset_mean.rename(
            columns={
                "policy_name": "uncertainty_policy",
                "final_dice": "uncertainty_mean_final_dice",
                "iterations_used": "uncertainty_mean_iterations_used",
                "n_permutations": "uncertainty_mean_n_permutations",
                "prompt_limit": "uncertainty_mean_prompt_limit",
            }
        )
        policy_nominated = nominated_starts.copy()
        policy_nominated["uncertainty_policy"] = str(policy_name)
        merged = policy_nominated.merge(
            uncertainty_policy,
            on=SUBSET_KEYS + ["uncertainty_policy", "start_image_id"],
            how="left",
            validate="many_to_one",
        )
        merged = merged.merge(
            subset_mean,
            on=SUBSET_KEYS + ["uncertainty_policy"],
            how="left",
            validate="many_to_one",
        )
        has_uncert = merged["nominated_final_dice"].notna()
        has_mean = merged["uncertainty_mean_final_dice"].notna()
        merged["matched_vs_uncertainty_mean"] = has_uncert & has_mean
        merged["coverage_status_secondary"] = np.select(
            [has_uncert & has_mean, ~has_uncert & has_mean, has_uncert & ~has_mean],
            ["matched", "missing_uncertainty_start", "missing_uncertainty_subset_mean"],
            default="missing_both",
        )
        merged["delta_final_dice"] = (
            merged["nominated_final_dice"] - merged["uncertainty_mean_final_dice"]
        )
        merged["delta_iterations_used_improvement"] = (
            merged["uncertainty_mean_iterations_used"] - merged["nominated_iterations_used"]
        )
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_base_analysis_table(primary_subset: pd.DataFrame, secondary_subset: pd.DataFrame) -> pd.DataFrame:
    """Build one flat table suitable as the default notebook analysis source."""
    if primary_subset.empty:
        return primary_subset.copy()

    base = primary_subset.copy()
    if secondary_subset.empty:
        return base

    secondary_cols = [
        "family",
        "task_id",
        "subset_index",
        "selector",
        "uncertainty_policy",
        "uncertainty_mean_final_dice",
        "uncertainty_mean_iterations_used",
        "uncertainty_mean_n_permutations",
        "uncertainty_mean_prompt_limit",
        "matched_vs_uncertainty_mean",
        "coverage_status_secondary",
    ]
    secondary_keep = [col for col in secondary_cols if col in secondary_subset.columns]
    if not secondary_keep:
        return base

    secondary = secondary_subset[secondary_keep].copy()
    join_keys = ["family", "task_id", "subset_index", "selector", "uncertainty_policy"]
    for key in join_keys:
        if key not in secondary.columns:
            return base

    merged = base.merge(secondary, on=join_keys, how="left", validate="one_to_one")
    if {"nominated_final_dice", "uncertainty_mean_final_dice"} <= set(merged.columns):
        merged["delta_final_dice_vs_uncertainty_mean"] = (
            merged["nominated_final_dice"] - merged["uncertainty_mean_final_dice"]
        )
    if {"nominated_iterations_used", "uncertainty_mean_iterations_used"} <= set(merged.columns):
        merged["delta_iterations_used_improvement_vs_uncertainty_mean"] = (
            merged["uncertainty_mean_iterations_used"] - merged["nominated_iterations_used"]
        )
    return merged


def _list_policy_names(
    *,
    repo_root: Path,
    procedure: str,
    ablation: str,
    dataset: str | None,
    mega_slicing: str | None,
) -> list[str]:
    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        dataset=dataset,
        mega_slicing=mega_slicing,
        filename="subset_support_images_summary.csv",
        allow_root_fallback=True,
    )
    return sorted(str(name) for name in df["policy_name"].dropna().unique().tolist())


def run_start_nomination_simulation(args: argparse.Namespace) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "figures" / "start_nomination_simulation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading uncertainty summaries: procedure={args.procedure} ablation={args.ablation} dataset={args.dataset or '<all>'}")
    uncertainty_df = load_planb_summaries(
        repo_root=repo_root,
        procedure=str(args.procedure),
        ablation=str(args.ablation),
        dataset=args.dataset,
        mega_slicing=args.mega_slicing,
        filename="subset_support_images_summary.csv",
        allow_root_fallback=True,
    )
    uncertainty_df = ensure_task_triple_columns(uncertainty_df)
    selected_policies = [str(name) for name in args.policy]
    uncertainty_df = uncertainty_df[uncertainty_df["policy_name"].isin(selected_policies)].copy()
    print(f"  {len(uncertainty_df)} rows for policies {selected_policies}")
    if uncertainty_df.empty:
        available = sorted(
            str(name) for name in load_planb_summaries(
                repo_root=repo_root,
                procedure=str(args.procedure),
                ablation=str(args.ablation),
                dataset=args.dataset,
                mega_slicing=args.mega_slicing,
                filename="subset_support_images_summary.csv",
                allow_root_fallback=True,
            )["policy_name"].dropna().unique().tolist()
        )
        raise ValueError(
            "No rows found for requested uncertainty policies. "
            f"Requested={selected_policies}, available={available}."
        )

    random_procedure = str(args.random_procedure) if args.random_procedure else str(args.procedure)
    random_ablation = str(args.random_ablation) if args.random_ablation else str(args.ablation)
    random_policy = str(args.random_policy)
    print(f"Loading random summaries: procedure={random_procedure} ablation={random_ablation} policy={random_policy}")
    random_df = load_planb_summaries(
        repo_root=repo_root,
        procedure=random_procedure,
        ablation=random_ablation,
        dataset=args.dataset,
        mega_slicing=args.mega_slicing,
        filename="subset_support_images_summary.csv",
        allow_root_fallback=True,
    )
    random_df = ensure_task_triple_columns(random_df)
    random_df = random_df[random_df["policy_name"] == random_policy].copy()
    print(f"  {len(random_df)} rows")
    if random_df.empty:
        available_random = sorted(
            str(name) for name in load_planb_summaries(
                repo_root=repo_root,
                procedure=random_procedure,
                ablation=random_ablation,
                dataset=args.dataset,
                mega_slicing=args.mega_slicing,
                filename="subset_support_images_summary.csv",
                allow_root_fallback=True,
            )["policy_name"].dropna().unique().tolist()
        )
        raise ValueError(
            f"No rows found for random policy '{random_policy}'. "
            f"Available={available_random}."
        )

    print("Summarising permutations...")
    uncertainty_perm = summarize_permutations(uncertainty_df)
    uncertainty_start_outcomes = build_start_conditioned_tables(uncertainty_perm)
    uncertainty_subset_outcomes = build_subset_mean_tables(uncertainty_perm)
    random_perm = summarize_permutations(random_df)
    random_subset_outcomes = build_subset_mean_tables(random_perm)
    random_subset_outcomes = random_subset_outcomes[random_subset_outcomes["policy_name"] == random_policy].copy()
    print(f"  {len(uncertainty_perm)} uncertainty permutations, {len(random_perm)} random permutations")

    subset_candidates = build_subset_candidate_table(uncertainty_df)
    if subset_candidates.empty:
        raise ValueError(
            "No subset candidates discovered from uncertainty summaries. "
            "Check procedure/ablation/policy/dataset filters."
        )
    print(f"  {len(subset_candidates)} subsets to embed")

    encoder_cfg = _load_encoder_cfg(Path(args.encoder_config_path), args.encoder_config_key)
    encoder_tag = _encoder_tag(encoder_cfg, args.encoder_config_key)
    embedding_cache_path = (
        Path(args.embedding_cache_path)
        if args.embedding_cache_path
        else out_dir / "embedding_cache.pkl"
    )
    embedding_cache = load_embedding_cache(embedding_cache_path)
    print(f"Building embeddings (encoder={encoder_cfg.get('type', '?')}, device={args.device}, cache={embedding_cache_path})...")

    from experiments.encoders.encoder_utils import build_encoder_from_cfg

    device = torch.device(str(args.device))
    encoder = build_encoder_from_cfg(encoder_cfg, device=device)
    subset_embedding_map, embedding_stats = build_subset_embedding_map(
        subset_candidates,
        encoder=encoder,
        encoder_tag=encoder_tag,
        data_split=str(args.data_split),
        device=device,
        dataset_seed=int(args.dataset_seed),
        embedding_cache=embedding_cache,
    )
    save_embedding_cache(embedding_cache_path, embedding_cache)
    print(f"  cache hits={embedding_stats['cache_hits']} misses={embedding_stats['cache_misses']}")

    selectors = [str(name) for name in args.selectors]
    print(f"Nominating starts (selectors={selectors})...")
    nominated_starts = nominate_starts(subset_embedding_map, selectors=selectors)
    if nominated_starts.empty:
        raise ValueError("No nominated starts were produced.")
    nominated_starts = nominated_starts.merge(
        subset_candidates[SUBSET_KEYS + ["task_name", "mega_task", "mega_label", "mega_slicing", "prompt_limit"]],
        on=SUBSET_KEYS,
        how="left",
        validate="many_to_one",
    )

    primary_subset = build_primary_subset_table(
        nominated_starts,
        uncertainty_start_outcomes,
        random_subset_outcomes,
        uncertainty_policy_names=selected_policies,
        random_policy_name=random_policy,
    )
    secondary_subset = build_secondary_subset_table(
        nominated_starts,
        uncertainty_start_outcomes,
        uncertainty_subset_outcomes,
        uncertainty_policy_names=selected_policies,
    )
    base_table = _build_base_analysis_table(primary_subset, secondary_subset)
    base_csv_name = str(args.base_csv_name).strip() if args.base_csv_name else "start_nomination_base.csv"
    if not base_csv_name:
        base_csv_name = "start_nomination_base.csv"

    outputs = {
        "base_csv": out_dir / base_csv_name,
        "run_metadata": out_dir / "run_metadata.json",
    }

    print(f"Writing output ({len(base_table)} rows): {outputs['base_csv']}")
    base_table.to_csv(outputs["base_csv"], index=False)

    metadata = {
        "procedure": str(args.procedure),
        "ablation": str(args.ablation),
        "uncertainty_policies": selected_policies,
        "random_procedure": random_procedure,
        "random_ablation": random_ablation,
        "random_policy": random_policy,
        "dataset": args.dataset,
        "mega_slicing": args.mega_slicing,
        "data_split": str(args.data_split),
        "dataset_seed": int(args.dataset_seed),
        "encoder_config_path": str(Path(args.encoder_config_path)),
        "encoder_config_key": args.encoder_config_key,
        "encoder_tag": encoder_tag,
        "selectors": selectors,
        "seed": int(args.seed),
        "cache_hits": int(embedding_stats["cache_hits"]),
        "cache_misses": int(embedding_stats["cache_misses"]),
        "embedding_cache_path": str(embedding_cache_path),
        "base_csv_name": base_csv_name,
    }
    outputs["run_metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate start-nominated uncertainty policies from existing Plan B summaries "
            "and evaluate primarily against random."
        )
    )
    parser.add_argument("--procedure", type=str, required=True)
    parser.add_argument("--ablation", type=str, required=True)
    parser.add_argument(
        "--policy",
        type=str,
        nargs="+",
        required=True,
        help="One or more uncertainty policy names to evaluate.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--mega-slicing", type=str, default=None)
    parser.add_argument("--random-procedure", type=str, default=None)
    parser.add_argument("--random-ablation", type=str, default=None)
    parser.add_argument("--random-policy", type=str, default="random")
    parser.add_argument("--encoder-config-path", type=Path, required=True)
    parser.add_argument(
        "--encoder-config-key",
        type=str,
        default=None,
        help="Optional top-level key inside encoder config YAML.",
    )
    parser.add_argument("--data-split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument(
        "--selectors",
        type=str,
        nargs="+",
        default=["closest_centroid", "medoid"],
        help=f"Selector names. Available: {list_selectors()}",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--embedding-cache-path", type=Path, default=None)
    parser.add_argument(
        "--base-csv-name",
        type=str,
        default="start_nomination_base.csv",
        help="Filename for the single flat notebook-friendly CSV output.",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="Print available policy names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_policies:
        repo_root = Path(__file__).resolve().parents[2]
        uncertainty_policies = _list_policy_names(
            repo_root=repo_root,
            procedure=str(args.procedure),
            ablation=str(args.ablation),
            dataset=args.dataset,
            mega_slicing=args.mega_slicing,
        )
        random_procedure = str(args.random_procedure) if args.random_procedure else str(args.procedure)
        random_ablation = str(args.random_ablation) if args.random_ablation else str(args.ablation)
        random_policies = _list_policy_names(
            repo_root=repo_root,
            procedure=random_procedure,
            ablation=random_ablation,
            dataset=args.dataset,
            mega_slicing=args.mega_slicing,
        )
        print("Uncertainty source policies:")
        for name in uncertainty_policies:
            print(f"- {name}")
        print("\nRandom source policies:")
        for name in random_policies:
            print(f"- {name}")
        return

    unknown_selectors = sorted(set(args.selectors) - set(list_selectors()))
    if unknown_selectors:
        raise ValueError(
            f"Unknown selectors: {unknown_selectors}. Available selectors: {list_selectors()}"
        )
    outputs = run_start_nomination_simulation(args)
    print("Wrote outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
