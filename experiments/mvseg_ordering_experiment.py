import os
import math

from experiments.dataset.mega_medical_dataset import MegaMedicalDataset
os.environ['NEURITE_BACKEND'] = 'pytorch'

# add MultiverSeg, UniverSeg and ScribblePrompt dependencies
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for dep in ["MultiverSeg", "UniverSeg", "ScribblePrompt"]:
    dep_path = PROJECT_ROOT / dep
    if str(dep_path) not in sys.path:
        sys.path.append(str(dep_path))

import neurite as ne
from typing import Any, Sequence, Optional, Dict
import torch
from .dataset.wbc_multiple_perms import WBCDataset
from multiverseg.models.sp_mvs import MultiverSeg
import yaml
from pylot.experiment.util import eval_config
from .score.dice_score import dice_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scribbleprompt.analysis.plot import show_points
from typing import Union, Dict
from .ordering import (
    OrderingConfig,
    RandomConfig,
    UncertaintyConfig,
    AdaptiveOrderingConfig,
    NonAdaptiveOrderingConfig,
)
DatasetType = Union[WBCDataset, MegaMedicalDataset]
DEFAULT_EVAL_STEP = 5


class MVSegOrderingExperiment():
    def __init__(
        self,
        support_dataset: DatasetType,
        prompt_generator: Any,
        prompt_iterations: int,
        commit_ground_truth: bool,
        dice_cutoff: float,
        interaction_protocol: str,
        script_dir: Path,
        ordering_config: OrderingConfig,
        should_visualize: bool = False,
        seed: int = 23,
        device: Optional[torch.device] = None,
        eval_fraction: Optional[float] = 0.1,
        eval_checkpoints: Optional[Sequence[int]] = None,
    ):
        if device is not None:
            resolved_device = torch.device(device)
        elif torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        else:
            resolved_device = torch.device("cpu")

        self.support_dataset = support_dataset
        self.prompt_generator = prompt_generator
        self.device = resolved_device
        self.model = MultiverSeg(version="v1", device=self.device)
        self.model.eval()
        self.prompt_iterations = prompt_iterations
        self.commit_ground_truth = commit_ground_truth
        self.ordering_config = ordering_config
        
        self.dice_cutoff = dice_cutoff
        self.seed = seed
        self.interaction_protocol = interaction_protocol
        self.results_dir = script_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.should_visualize = should_visualize

        if isinstance(self.ordering_config, UncertaintyConfig):
            if self.ordering_config.tyche_sampler is None:
                raise RuntimeError("UncertaintyConfig requires a TycheAugs sampler, but none was provided.")

        if eval_fraction is not None:
            if not (0 < eval_fraction <= 1):
                raise ValueError("eval_fraction must be in the interval (0, 1].")
            self.eval_fraction = float(eval_fraction)
        else:
            self.eval_fraction = None

        if eval_checkpoints is not None:
            checkpoints = sorted({int(k) for k in eval_checkpoints if int(k) > 0})
            if not checkpoints:
                raise ValueError("eval_checkpoints must contain at least one positive integer.")
            self.eval_checkpoints = checkpoints
        else:
            self.eval_checkpoints = None

        # set seeds
        np.random.seed(seed)
        random.seed(seed) 
        torch.manual_seed(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


    def run_permutations(self):
        train_indices = list(self.support_dataset.get_data_indices())
        all_iterations = []
        all_images = []
        all_eval_iterations = []
        all_eval_images = []

        support_indices = train_indices
        eval_indices: list[int] = []
        if self.eval_fraction is not None:
            eval_count = max(1, math.ceil(len(train_indices) * self.eval_fraction))
            rng_eval = np.random.default_rng(self.seed)
            eval_indices = rng_eval.choice(train_indices, size=eval_count, replace=False).tolist()
            eval_index_set = set(eval_indices)
            support_indices = [idx for idx in train_indices if idx not in eval_index_set]
            if not support_indices:
                raise ValueError(
                    "Evaluation split consumed all available support samples. "
                    "Reduce eval_fraction or provide a larger dataset."
                )

        # Branch: adaptive (online) vs non-adaptive (precomputed) orderings.
        if isinstance(self.ordering_config, AdaptiveOrderingConfig):
            (
                all_iterations,
                all_images,
                all_eval_iterations,
                all_eval_images,
            ) = self._run_uncertainty_permutations(
                support_indices=support_indices,
                eval_indices=eval_indices,
            )
            self._write_aggregate_results(
                frames=all_iterations,
                output_path=self.results_dir / "support_images_iterations.csv",
            )

            self._write_aggregate_results(
                frames=all_images,
                output_path=self.results_dir / "support_images_summary.csv",
            )

            if all_eval_iterations:
                self._write_aggregate_results(
                    frames=all_eval_iterations,
                    output_path=self.results_dir / "eval_iterations.csv",
                )
            if all_eval_images:
                self._write_aggregate_results(
                    frames=all_eval_images,
                    output_path=self.results_dir / "eval_image_summary.csv",
                )
            return

        orderings = self.ordering_config.get_orderings(
            support_dataset=self.support_dataset,
            candidate_indices=support_indices,
        )
        ordering_labels = self.ordering_config.get_ordering_labels()
        ordering_seeds = self.ordering_config.get_ordering_seeds()
        ordering_labels = list(ordering_labels)
        ordering_seeds = list(ordering_seeds)
        if len(ordering_labels) != len(orderings):
            raise ValueError("Ordering labels length does not match generated orderings.")
        if len(ordering_seeds) != len(orderings):
            raise ValueError("Ordering seeds length does not match generated orderings.")

        ordering_triplets = list(zip(ordering_labels, orderings, ordering_seeds))

        for permutation_index, shuffled_indices, perm_gen_seed in ordering_triplets:
            print(f"Doing Perm {permutation_index}...")
            data = [self.support_dataset.get_item_by_data_index(index) for index in shuffled_indices]
            support_images, support_labels = zip(*data)
            support_images = torch.stack(support_images).to(self.device)
            support_labels = torch.stack(support_labels).to(self.device)
            seed_folder_dir = self.results_dir / f"Perm_Seed_{permutation_index}"
            seed_folder_dir.mkdir(exist_ok=True)

            # Run the full support pass once to build the entire context and logs
            (
                per_iteration_records,
                per_image_records,
                full_context_images,
                full_context_labels,
            ) = self.run_seq_multiverseg(
                support_images=support_images,
                support_labels=support_labels,
                image_ids=shuffled_indices,
                perm_gen_seed=perm_gen_seed,
                ordering_index=permutation_index,
                seed_folder_dir=seed_folder_dir,
            )
            per_iteration_records.to_csv(seed_folder_dir / "per_iteration_records.csv", index=False)
            per_image_records.to_csv(seed_folder_dir / "per_image_records.csv", index=False)
            all_iterations.append(per_iteration_records)
            all_images.append(per_image_records)

            if self.eval_fraction is not None and eval_indices:
                eval_iteration_df, eval_image_df = self._evaluate_heldout_over_context(
                    eval_indices=eval_indices,
                    perm_gen_seed=perm_gen_seed,
                    permutation_index=permutation_index,
                    seed_folder_dir=seed_folder_dir,
                    full_context_images=full_context_images,
                    full_context_labels=full_context_labels,
                    context_steps=self.eval_checkpoints,
                )
                eval_dir = seed_folder_dir / "eval"
                eval_dir.mkdir(exist_ok=True)
                eval_iteration_df.to_csv(eval_dir / "eval_iteration_records.csv", index=False)
                eval_image_df.to_csv(eval_dir / "eval_image_records.csv", index=False)
                all_eval_iterations.append(eval_iteration_df)
                all_eval_images.append(eval_image_df)

        self._write_aggregate_results(
            frames=all_iterations,
            output_path=self.results_dir / "support_images_iterations.csv",
        )

        self._write_aggregate_results(
            frames=all_images,
            output_path=self.results_dir / "support_images_summary.csv",
        )

        if all_eval_iterations:
            self._write_aggregate_results(
                frames=all_eval_iterations,
                output_path=self.results_dir / "eval_iterations.csv",
            )
        if all_eval_images:
            self._write_aggregate_results(
                frames=all_eval_images,
                output_path=self.results_dir / "eval_image_summary.csv",
            )

    def _write_aggregate_results(self, frames, output_path: Path) -> None:
        if not frames:
            return
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)

    def _append_iteration_record(
        self,
        rows,
        perm_gen_seed: int,
        ordering_index: int,
        image_index: int,
        image_id: Any,
        iteration: int,
        score: float,
        pos_clicks: int,
        neg_clicks: int,
        context_size: int
    ) -> None:
        rows.append({
            "policy_name": self.ordering_config.name,
            "experiment_seed": self.seed,
            "perm_gen_seed": perm_gen_seed,
            "permutation_index": ordering_index,
            "image_index": image_index,
            "image_id": image_id,
            "iteration": iteration,
            "commit_ground_truth": self.commit_ground_truth,
            "dice_cutoff": self.dice_cutoff,
            "pos_clicks": pos_clicks,
            "neg_clicks": neg_clicks,
            "score": score,
            "prompt_limit": self.prompt_iterations,
            "context_size": context_size
        })

    def _append_image_summary_record(
        self,
        image_summary_rows,
        perm_gen_seed: int,
        ordering_index: int,
        image_index: int,
        image_id: Any,
        initial_dice: float,
        final_dice: float,
        dice_at_goal: float,
        iterations_to_goal: int,
        reached_cutoff: bool,
        context_size: int
    ) -> None:
        image_summary_rows.append({
            "policy_name": self.ordering_config.name,
            "experiment_seed": self.seed,
            "perm_gen_seed": perm_gen_seed,
            "permutation_index": ordering_index,
            "image_index": image_index,
            "image_id": image_id,
            "initial_dice": initial_dice,
            "final_dice": final_dice,
            "dice_at_goal": dice_at_goal,
            "iterations_to_goal": iterations_to_goal,
            "reached_cutoff": reached_cutoff,
            "commit_type": "ground_truth" if self.commit_ground_truth else "prediction",
            "protocol": self.interaction_protocol,
            "dice_cutoff": self.dice_cutoff,
            "prompt_limit": self.prompt_iterations,
            "context_size": context_size
        })

    def _visualize_data(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        prediction: torch.Tensor,
        annotations,
        seed_folder_dir: Path,
        image_index: int,
        iteration: int,
    ) -> None:

        fig, ax = ne.plot.slices([image.detach().cpu(), label.detach().cpu(), (prediction > 0).detach().cpu()], width=10,
                                  titles=['Image', 'Label', 'Prediction'])
        point_coords = annotations.get('point_coords')
        point_labels = annotations.get('point_labels')
        if point_coords is not None and point_labels is not None:
            show_points(point_coords.detach().cpu(), point_labels.detach().cpu(), ax=ax[0])
        figure_dir = seed_folder_dir / "Prediction Figures"
        figure_dir.mkdir(exist_ok=True)
        fig.savefig(figure_dir / f"Image_{image_index}_prediction_{iteration}.png")
        plt.close()

    def _update_context(
        self,
        context_images: torch.Tensor,
        context_labels: torch.Tensor,
        image: torch.Tensor,
        label: torch.Tensor,
        prediction: torch.Tensor,
    ):
        binary_prediction = (prediction > 0).float()
        mask_to_commit = label[None, ...] if self.commit_ground_truth else binary_prediction
        # # B x n x 1 x H x W
        if context_images is None:
            context_images = image[None, None, ...]
            context_labels = mask_to_commit[None, ...]
        else:
            context_images = torch.cat([context_images, image[None, None, ...]], dim=1)
            context_labels = torch.cat([context_labels, mask_to_commit[None, ...]], dim=1)

        return context_images, context_labels

    def _evaluate_heldout_over_context(
        self,
        eval_indices: Sequence[int],
        perm_gen_seed: int,
        permutation_index: int,
        seed_folder_dir: Path,
        full_context_images: Optional[torch.Tensor],
        full_context_labels: Optional[torch.Tensor],
        context_steps: Optional[Sequence[int]] = None,
    ):
        # Load held-out data once
        eval_data = [self.support_dataset.get_item_by_data_index(index) for index in eval_indices]
        eval_images, eval_labels = zip(*eval_data)
        eval_images = torch.stack(eval_images).to(self.device)
        eval_labels = torch.stack(eval_labels).to(self.device)

        eval_iter_all = []
        eval_img_all = []

        # Context size 0 (no support committed)
        print("Eval for slice 0")
        zero_context_iteration_results, zero_context_summary_results = self.run_seq_multiverseg_eval(
            test_images=eval_images,
            test_labels=eval_labels,
            image_ids=eval_indices,
            perm_gen_seed=perm_gen_seed,
            ordering_index=permutation_index,
            seed_folder_dir=seed_folder_dir,
            context_images=None,
            context_labels=None,
        )
        eval_iter_all.append(zero_context_iteration_results)
        eval_img_all.append(zero_context_summary_results)

        context_size = 0 if full_context_images is None else len(full_context_images[0])
        eval_steps = context_steps if context_steps is not None \
            else list(range(DEFAULT_EVAL_STEP, context_size + 1, DEFAULT_EVAL_STEP))

        for k in eval_steps:
            print(f"Eval for slice {k}")
            if k > context_size:
                break
            sliced_context_images = full_context_images[:, :k, ...]
            sliced_context_labels = full_context_labels[:, :k, ...]
            k_context_iteration_results, k_context_summary_results = self.run_seq_multiverseg_eval(
                test_images=eval_images,
                test_labels=eval_labels,
                image_ids=eval_indices,
                perm_gen_seed=perm_gen_seed,
                ordering_index=permutation_index,
                seed_folder_dir=seed_folder_dir,
                context_images=sliced_context_images,
                context_labels=sliced_context_labels,
            )
            eval_iter_all.append(k_context_iteration_results)
            eval_img_all.append(k_context_summary_results)

        eval_iteration_all_df = pd.concat(eval_iter_all, ignore_index=True)
        eval_image_all_df = pd.concat(eval_img_all, ignore_index=True)
        return eval_iteration_all_df, eval_image_all_df

    def _run_prompt_loop(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        image_index: int,
        image_id: Any,
        context_images: torch.Tensor,
        context_labels: torch.Tensor,
        yhat: torch.Tensor,
        rows,
        perm_gen_seed: int,
        ordering_index: int,
        seed_folder_dir: Path,
    ):
        # Score at goal
        goal_yhat = None
        score_at_goal = None
        iterations_needed = None

        prompts = None
        context_size = 0 if context_images is None else len(context_images[0])

        for iteration in range(self.prompt_iterations):
            if iteration == 0:
                prompts = self.prompt_generator(image[None], label[None])
            else:
                prompts = self.prompt_generator.subsequent_prompt(
                    mask_pred=yhat,
                    prev_input=prompts,
                    new_prompt=True,
                )

            annotations = {k: prompts.get(k) for k in ['point_coords', 'point_labels', 'mask_input', 'scribbles', 'box']}
            yhat = self.model.predict(image[None], context_images, context_labels, **annotations, return_logits=True).to(self.device)

            if self.should_visualize:
                self._visualize_data(
                    image=image,
                    label=label,
                    prediction=yhat,
                    annotations=annotations,
                    seed_folder_dir=seed_folder_dir,
                    image_index=image_index,
                    iteration=iteration,
                )

            score = dice_score((yhat > 0).float(), label[None, ...])
            score_value = float(score.item())

            if prompts.get('point_labels', None) is not None:
                pos_clicks = prompts.get('point_labels').sum().item()
                neg_clicks = prompts.get('point_labels').shape[1] - pos_clicks
            else:
                pos_clicks = neg_clicks = 0

            self._append_iteration_record(
                rows=rows,
                perm_gen_seed=perm_gen_seed,
                ordering_index=ordering_index,
                image_index=image_index,
                image_id=image_id,
                iteration=iteration,
                score=score_value,
                pos_clicks=pos_clicks,
                neg_clicks=neg_clicks,
                context_size=context_size
            )

            if score_value >= self.dice_cutoff and iterations_needed is None:
                iterations_needed = iteration + 1
                goal_yhat = yhat
                score_at_goal = score_value

        if iterations_needed is None:
            iterations_needed = self.prompt_iterations
            goal_yhat = yhat
            score_at_goal = score_value
        return score_value, iterations_needed, goal_yhat, score_at_goal

    def _get_curriculum_config(self) -> UncertaintyConfig:
        if not isinstance(self.ordering_config, UncertaintyConfig):
            raise RuntimeError("Curriculum configuration is not set.")
        return self.ordering_config

    def _run_uncertainty_permutations(
        self,
        support_indices: Sequence[int],
        eval_indices: Sequence[int],
    ):
        """
        Adaptive ordering: select next image based on uncertainty scores that use the current context.
        """
        if not isinstance(self.ordering_config, UncertaintyConfig):
            raise RuntimeError("UncertaintyConfig not set for adaptive ordering.")

        all_iterations = []
        all_images = []
        all_eval_iterations = []
        all_eval_images = []

        start_indices = self.ordering_config.get_start_positions(support_indices)
        total_runs = len(start_indices)
        for run_idx, start_index in enumerate(start_indices):
            print(f"[uncertainty] Run {run_idx+1}/{total_runs} start_index={start_index}")
            perm_gen_seed = self.seed + run_idx
            seed_folder_dir = self.results_dir / f"Perm_Seed_{start_index}"
            seed_folder_dir.mkdir(exist_ok=True)

            remaining = list(support_indices)
            # Deterministic start
            if start_index not in remaining:
                raise ValueError(f"Start index {start_index} not in support_indices.")
            ordering_sequence: list[int] = []
            context_images = None
            context_labels = None
            rows = []
            image_summary_rows = []

            rng = np.random.default_rng(perm_gen_seed)
            current_index = start_index

            while remaining:
                remaining.remove(current_index)
                image, label = self.support_dataset.get_item_by_data_index(current_index)
                image = image.to(self.device)
                label = label.to(self.device)
                ordering_sequence.append(current_index)
                context_size = 0 if context_images is None else len(context_images[0])

                # Initial prediction
                yhat = self.model.predict(image[None], context_images, context_labels, return_logits=True).to(self.device)
                score = dice_score((yhat > 0).float(), label[None, ...])
                initial_dice = float(score.item())

                self._append_iteration_record(
                    rows=rows,
                    perm_gen_seed=perm_gen_seed,
                    ordering_index=start_index,
                    image_index=len(ordering_sequence) - 1,
                    image_id=current_index,
                    iteration=-1,
                    score=initial_dice,
                    pos_clicks=0,
                    neg_clicks=0,
                    context_size=context_size,
                )

                if initial_dice >= self.dice_cutoff:
                    iterations_used = 0
                    dice_at_goal = initial_dice
                    final_dice, _, _, _ = self._run_prompt_loop(
                        image=image,
                        label=label,
                        image_index=len(ordering_sequence) - 1,
                        image_id=current_index,
                        context_images=context_images,
                        context_labels=context_labels,
                        yhat=yhat,
                        rows=rows,
                        perm_gen_seed=perm_gen_seed,
                        ordering_index=start_index,
                        seed_folder_dir=seed_folder_dir,
                    )
                else:
                    final_dice, iterations_used, yhat, dice_at_goal = self._run_prompt_loop(
                        image=image,
                        label=label,
                        image_index=len(ordering_sequence) - 1,
                        image_id=current_index,
                        context_images=context_images,
                        context_labels=context_labels,
                        yhat=yhat,
                        rows=rows,
                        perm_gen_seed=perm_gen_seed,
                        ordering_index=start_index,
                        seed_folder_dir=seed_folder_dir,
                    )

                self._append_image_summary_record(
                    image_summary_rows=image_summary_rows,
                    perm_gen_seed=perm_gen_seed,
                    ordering_index=start_index,
                    image_index=len(ordering_sequence) - 1,
                    image_id=current_index,
                    initial_dice=initial_dice,
                    final_dice=final_dice,
                    dice_at_goal=dice_at_goal,
                    iterations_used=iterations_used,
                    reached_cutoff=dice_at_goal >= self.dice_cutoff,
                    context_size=context_size,
                )

                # Commit to context
                context_images, context_labels = self._update_context(
                    context_images=context_images,
                    context_labels=context_labels,
                    image=image,
                    label=label,
                    prediction=yhat,
                )

                # Select next based on uncertainty
                if remaining:
                    current_index = self.ordering_config.select_index_by_uncertainty(
                        candidate_indices=remaining,
                        support_dataset=self.support_dataset,
                        model=self.model,
                        device=self.device,
                        context_images=context_images,
                        context_labels=context_labels,
                    )

            per_iteration_records = pd.DataFrame.from_records(rows)
            per_image_records = pd.DataFrame.from_records(image_summary_rows)
            per_iteration_records.to_csv(seed_folder_dir / "per_iteration_records.csv", index=False)
            per_image_records.to_csv(seed_folder_dir / "per_image_records.csv", index=False)
            all_iterations.append(per_iteration_records)
            all_images.append(per_image_records)

            if self.eval_fraction is not None and eval_indices:
                eval_iteration_df, eval_image_df = self._evaluate_heldout_over_context(
                    eval_indices=eval_indices,
                    perm_gen_seed=perm_gen_seed,
                    permutation_index=start_index,
                    seed_folder_dir=seed_folder_dir,
                    full_context_images=context_images,
                    full_context_labels=context_labels,
                    context_steps=self.eval_checkpoints,
                )
                eval_dir = seed_folder_dir / "eval"
                eval_dir.mkdir(exist_ok=True)
                eval_iteration_df.to_csv(eval_dir / "eval_iteration_records.csv", index=False)
                eval_image_df.to_csv(eval_dir / "eval_image_records.csv", index=False)
                all_eval_iterations.append(eval_iteration_df)
                all_eval_images.append(eval_image_df)

        return all_iterations, all_images, all_eval_iterations, all_eval_images

    @torch.inference_mode()
    def _run_seq_common(
        self,
        images,
        labels,
        image_ids,
        perm_gen_seed,
        ordering_index,
        seed_folder_dir,
        context_images=None,
        context_labels=None,
        update_context=False,
        return_context=False,
    ):
        # N x C x H x W for the provided images and labels
        rows = []
        image_summary_rows = []
        assert(images.size(0) == labels.size(0))
        assert len(image_ids) == images.size(0)

        for index in range(images.size(0)):
            print(f"Doing Image {index+1}/{images.size(0)}...")

            # Image and Label: C x H x W
            image = images[index]
            label = labels[index]
            image_id = image_ids[index]

            # Initial Dice Input is B x C x H x W and output is  # B x C x H x W
            yhat = self.model.predict(image[None], context_images, context_labels, return_logits=True).to(self.device)
            score = dice_score((yhat > 0).float(), label[None, ...])
            initial_dice = float(score.item())
            context_size = 0 if context_images is None else len(context_images[0])

            if self.should_visualize:
                self._visualize_data(
                    image=image,
                    label=label,
                    prediction=yhat,
                    annotations={},
                    seed_folder_dir=seed_folder_dir,
                    image_index=index,
                    iteration=-1,
                )

            self._append_iteration_record(
                rows=rows,
                perm_gen_seed=perm_gen_seed,
                ordering_index=ordering_index,
                image_index=index,
                image_id=image_id,
                iteration=-1,
                score=initial_dice,
                pos_clicks=0,
                neg_clicks=0,
                context_size=context_size
            )

            if initial_dice >= self.dice_cutoff:
                iterations_used = 0
                dice_at_goal = initial_dice
                final_dice, _, _, _ = self._run_prompt_loop(
                    image=image,
                    label=label,
                    image_index=index,
                    image_id=image_id,
                    context_images=context_images,
                    context_labels=context_labels,
                    yhat=yhat,
                    rows=rows,
                    perm_gen_seed=perm_gen_seed,
                    ordering_index=ordering_index,
                    seed_folder_dir=seed_folder_dir,
                )
            else:
                final_dice, iterations_used, yhat, dice_at_goal = self._run_prompt_loop(
                    image=image,
                    label=label,
                    image_index=index,
                    image_id=image_id,
                    context_images=context_images,
                    context_labels=context_labels,
                    yhat=yhat,
                    rows=rows,
                    perm_gen_seed=perm_gen_seed,
                    ordering_index=ordering_index,
                    seed_folder_dir=seed_folder_dir,
                )
            self._append_image_summary_record(
                image_summary_rows=image_summary_rows,
                perm_gen_seed=perm_gen_seed,
                ordering_index=ordering_index,
                image_index=index,
                image_id=image_id,
                initial_dice=initial_dice,
                final_dice=final_dice,
                dice_at_goal=dice_at_goal,
                iterations_to_goal=iterations_used,
                reached_cutoff=dice_at_goal >= self.dice_cutoff,
                context_size=context_size,
            )

            if update_context:
                context_images, context_labels = self._update_context(
                    context_images=context_images,
                    context_labels=context_labels,
                    image=image,
                    label=label,
                    prediction=yhat,
                )
        iteration_df = pd.DataFrame.from_records(rows)
        image_df = pd.DataFrame.from_records(image_summary_rows)

        if return_context:
            return (
                iteration_df,
                image_df,
                context_images,
                context_labels,
            )
        return iteration_df, image_df
    
    def run_seq_multiverseg_eval(
        self,
        test_images,
        test_labels,
        image_ids: Sequence[int],
        perm_gen_seed: int,
        ordering_index,
        seed_folder_dir,
        context_images=None,
        context_labels=None,
    ):
        return self._run_seq_common(
            images=test_images,
            labels=test_labels,
            image_ids=image_ids,
            perm_gen_seed=perm_gen_seed,
            ordering_index=ordering_index,
            seed_folder_dir=seed_folder_dir,
            context_images=context_images,
            context_labels=context_labels,
            update_context=False,
        )

    def run_seq_multiverseg(
        self,
        support_images,
        support_labels,
        image_ids: Sequence[int],
        perm_gen_seed: int,
        ordering_index,
        seed_folder_dir,
    ):
        return self._run_seq_common(
            images=support_images,
            labels=support_labels,
            image_ids=image_ids,
            perm_gen_seed=perm_gen_seed,
            ordering_index=ordering_index,
            seed_folder_dir=seed_folder_dir,
            context_images=None,
            context_labels=None,
            update_context=True,
            return_context=True,
        )

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    train_split = 0.6
    d_support = WBCDataset('JTSC', split='support', label='nucleus', support_frac=train_split, testing_data_size=15)
    with open(script_dir / "prompt_generator_configs/click_prompt_generator.yml", "r") as f:
        cfg = yaml.safe_load(f)
    prompt_generator_config = cfg['click_generator']
    prompt_generator =  eval_config(cfg)['click_generator']
    protocol_desc = (
        f"{prompt_generator_config.get('init_pos_click', 0)}_init_pos,"
        f"{prompt_generator_config.get('init_neg_click', 0)}_init_neg,"
        f"{prompt_generator_config.get('correction_clicks', 0)}_corrections"
    )
    experiment = MVSegOrderingExperiment(
        support_dataset=d_support, 
        prompt_generator=prompt_generator, 
        prompt_iterations=20, 
        commit_ground_truth=False, 
        dice_cutoff=0.9, 
        interaction_protocol=f"{protocol_desc}",
        script_dir=script_dir,
        should_visualize=False,
        ordering_config=RandomConfig(seed=23, permutations=5),
    )
    experiment.run_permutations()

    # experiment_number = 1
    # experiment = MVSegOrderingExperiment(
    #     dataset=d_support, 
    #     prompt_generator=prompt_generator, 
    #     prompt_iterations=5,
    #     commit_ground_truth=True,
    #     dice_cutoff=0.9,
    #     interaction_protocol=str(experiment_number),
    #     ordering_config=RandomConfig(seed=23, permutations=10),
    # )
        
