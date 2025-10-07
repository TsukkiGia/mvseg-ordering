import os
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
from typing import Any, Sequence, Optional
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


class MVSegOrderingExperiment():
    def __init__(
        self,
        support_dataset: WBCDataset,
        prompt_generator: Any,
        prompt_iterations: int,
        commit_ground_truth: bool,
        permutations: int,
        dice_cutoff: float,
        interaction_protocol: str,
        script_dir: Path,
        should_visualize: bool = False,
        seed: int = 23
    ):
        
        self.support_dataset = support_dataset
        self.prompt_generator = prompt_generator
        self.model = MultiverSeg(version="v1")
        self.prompt_iterations = prompt_iterations
        self.commit_ground_truth = commit_ground_truth
        self.permutations = permutations
        self.dice_cutoff = dice_cutoff
        self.seed = seed
        self.interaction_protocol = interaction_protocol
        self.results_dir = script_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.should_visualize = should_visualize

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
        for permutation_index in range(self.permutations):
            print(f"Doing Perm {permutation_index}...")
            perm_gen_seed = self.seed + permutation_index
            rng = np.random.default_rng(perm_gen_seed)
            shuffled_indices = rng.permutation(train_indices).tolist()
            shuffled_data = [self.support_dataset.get_item_by_data_index(index) for index in shuffled_indices]
            support_images, support_labels = zip(*shuffled_data)
            support_images = torch.stack(support_images).to("cpu")
            support_labels = torch.stack(support_labels).to("cpu")
            seed_folder_dir =  self.results_dir / f"Perm_Seed_{permutation_index}"
            seed_folder_dir.mkdir(exist_ok=True)

            # Run the full support pass once to build the entire context and logs
            per_iteration_records, per_image_records = self.run_seq_multiverseg(
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
                
        self._write_aggregate_results(
            frames=all_iterations,
            output_path=self.results_dir / "support_images_iterations.csv",
        )

        self._write_aggregate_results(
            frames=all_images,
            output_path=self.results_dir / "support_images_summary.csv",
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
        iterations_used: int,
        reached_cutoff: bool,
        context_size: int
    ) -> None:
        image_summary_rows.append({
            "experiment_seed": self.seed,
            "perm_gen_seed": perm_gen_seed,
            "permutation_index": ordering_index,
            "image_index": image_index,
            "image_id": image_id,
            "initial_dice": initial_dice,
            "final_dice": final_dice,
            "iterations_used": iterations_used,
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

        fig, ax = ne.plot.slices([image.cpu(), label.cpu(), prediction > 0], width=10,
                                  titles=['Image', 'Label', 'Prediction'])
        point_coords = annotations.get('point_coords')
        point_labels = annotations.get('point_labels')
        if point_coords is not None and point_labels is not None:
            show_points(point_coords.cpu(), point_labels.cpu(), ax=ax[0])
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
    ):
        # Load held-out data once
        eval_data = [self.support_dataset.get_item_by_data_index(index) for index in eval_indices]
        eval_images, eval_labels = zip(*eval_data)
        eval_images = torch.stack(eval_images).to("cpu")
        eval_labels = torch.stack(eval_labels).to("cpu")

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
        for k in range(1, context_size + 1):
            print(f"Eval for slice {k}")
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
        score_value = float(dice_score((yhat > 0).float(), label[None, ...]).item())
        iterations_used = 0
        prompts = None
        context_size = 0 if context_images is None else len(context_images[0])

        for iteration in range(self.prompt_iterations):
            iterations_used = iteration + 1
            if iteration == 0:
                prompts = self.prompt_generator(image[None], label[None])
            else:
                prompts = self.prompt_generator.subsequent_prompt(
                    mask_pred=yhat,
                    prev_input=prompts,
                    new_prompt=True,
                )

            annotations = {k: prompts.get(k) for k in ['point_coords', 'point_labels', 'mask_input', 'scribbles', 'box']}
            yhat = self.model.predict(image[None], context_images, context_labels, **annotations, return_logits=True).to('cpu')

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

            if score_value >= self.dice_cutoff:
                break

        return score_value, iterations_used, yhat

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

            # Initial Dice
            yhat = self.model.predict(image[None], context_images, context_labels, return_logits=True).to('cpu')
            score = dice_score((yhat > 0).float(), label[None, ...])
            initial_dice = float(score.item())
            context_size = 0 if context_images is None else len(context_images[0])

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
                self._append_image_summary_record(
                    image_summary_rows=image_summary_rows,
                    perm_gen_seed=perm_gen_seed,
                    ordering_index=ordering_index,
                    image_index=index,
                    image_id=image_id,
                    initial_dice=initial_dice,
                    final_dice=initial_dice,
                    iterations_used=0,
                    reached_cutoff=True,
                    context_size=context_size
                )
            else:
                final_dice, iterations_used, yhat = self._run_prompt_loop(
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
                    iterations_used=iterations_used,
                    reached_cutoff=final_dice >= self.dice_cutoff,
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
        return (
            pd.DataFrame.from_records(rows),
            pd.DataFrame.from_records(image_summary_rows),
        )
    
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
        permutations=5, 
        dice_cutoff=0.9, 
        interaction_protocol=f"{protocol_desc}",
        script_dir=script_dir,
        should_visualize=False
    )
    experiment.run_permutations()

    # experiment_number = 1
    # experiment = MVSegOrderingExperiment(
    #     dataset=d_support, 
    #     prompt_generator=prompt_generator, 
    #     prompt_iterations=5, 
    #     commit_ground_truth=True, 
    #     permutations=10, 
    #     dice_cutoff=0.9, 
    #     interaction_protocol=str(experiment_number))
        
