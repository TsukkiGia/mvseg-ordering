import os
os.environ['NEURITE_BACKEND'] = 'pytorch'

# add MultiverSeg, UniverSeg and ScribblePrompt dependencies
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for dep in ["MultiverSeg", "UniverSeg", "ScribblePrompt"]:
    dep_path = PROJECT_ROOT / dep
    if str(dep_path) not in sys.path:
        sys.path.append(str(dep_path))

import neurite as ne
from typing import Any
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
    def __init__(self, 
                 support_dataset: WBCDataset,
                 test_dataset: WBCDataset, 
                 prompt_generator: Any, 
                 prompt_iterations: int, 
                 commit_ground_truth: bool,
                 permutations: int,
                 dice_cutoff: float,
                 interaction_protocol: str,
                 experiment_number: int,
                 script_dir: Path,
                 should_visualize: bool = False,
                 seed: int = 23
                 ):
        
        self.support_dataset = support_dataset
        self.test_dataset = test_dataset
        self.prompt_generator = prompt_generator
        self.model = MultiverSeg(version="v0")
        self.prompt_iterations = prompt_iterations
        self.commit_ground_truth = commit_ground_truth
        self.permutations = permutations
        self.dice_cutoff = dice_cutoff
        self.seed = seed
        self.interaction_protocol = interaction_protocol
        results_dir = script_dir / "results"
        results_dir.mkdir(exist_ok=True)
        self.experiment_folder = results_dir / f"Experiment_{experiment_number}"
        self.experiment_folder.mkdir(exist_ok=True)
        self.experiment_number = experiment_number
        self.should_visualize = should_visualize
        np.random.seed(seed)


    def run_permutations(self):
        base_indices = self.support_dataset.get_data_indices()
        all_iterations = []
        all_images = []
        for permutation_index in range(self.permutations):
            print(f"Doing Perm {permutation_index}...")
            rng = np.random.default_rng(permutation_index)
            shuffled_indices = rng.permutation(base_indices).tolist()
            shuffled_data = [self.support_dataset.get_item_by_data_index(index) for index in shuffled_indices]
            support_images, support_labels = zip(*shuffled_data)
            support_images = torch.stack(support_images).to("cpu")
            support_labels = torch.stack(support_labels).to("cpu")
            seed_folder_dir =  self.experiment_folder / f"Perm_Seed_{permutation_index}"
            seed_folder_dir.mkdir(exist_ok=True)
            per_iteration_records, per_image_records = self.run_seq_multiverseg(support_images, support_labels, permutation_index, seed_folder_dir)
            per_iteration_records.to_csv(seed_folder_dir / "per_iteration_records.csv", index=False)
            per_image_records.to_csv(seed_folder_dir / "per_image_records.csv", index=False)
            all_iterations.append(per_iteration_records)
            all_images.append(per_image_records)
        all_iteration_results = pd.concat(all_iterations, ignore_index=True)
        all_image_results = pd.concat(all_images, ignore_index=True)
        all_iteration_results.to_csv(self.experiment_folder / "all_iteration_results.csv", index=False)
        all_image_results.to_csv(self.experiment_folder / "all_image_results.csv", index=False)
    
    def _run_seq_common(self, images, labels, ordering_index, seed_folder_dir, *, update_context: bool):
        # N x C x H x W for the provided images and labels
        rows = []
        image_summary_rows = []
        assert(images.size(0) == labels.size(0))
        context_images = None
        context_labels = None

        for index in range(images.size(0)):
            print(f"Doing Image {index+1}/{images.size(0)}...")
            # Image and Label: C x H x W
            image = images[index]
            label = labels[index]

            # For each image, we are doing max 20 iterations to get to 90 Dice
            for iteration in range(self.prompt_iterations):
                if iteration == 0:
                    prompts = self.prompt_generator(image[None], label[None])
                else:
                    prompts = self.prompt_generator.subsequent_prompt(
                            mask_pred=yhat, # shape: (B, C, H, W)
                            prev_input=prompts,
                            new_prompt=True
                    )
                
                annotations = {k:prompts.get(k) for k in ['point_coords', 'point_labels', 'mask_input', 'scribbles', 'box']}
                yhat = self.model.predict(image[None], context_images, context_labels, **annotations, return_logits=True).to('cpu')
                
                # visualize result
                if self.should_visualize:
                    fig, ax = ne.plot.slices([image.cpu(), label.cpu(), yhat > 0], width=10, 
                    titles=['Image', 'Label', 'Prediction'])
                    point_coords = annotations.get('point_coords')
                    point_labels = annotations.get('point_labels')
                    if point_coords is not None and point_labels is not None:
                        show_points(point_coords.cpu(), point_labels.cpu(), ax=ax[0])
                    figure_dir = seed_folder_dir / "Prediction Figures"
                    figure_dir.mkdir(exist_ok=True)
                    fig.savefig(figure_dir / f"Image_{index}_prediction_{iteration}.png")
                    plt.close()

                # get score for yhat
                score = dice_score((yhat > 0).float(), label[None, ...])
                if iteration == 0:
                    initial_dice = float(score.item())

                # get interaction stats
                if prompts.get('point_labels', None) is not None:
                    pos_clicks = prompts.get('point_labels').sum().item()
                    neg_clicks = prompts.get('point_labels').shape[1] - pos_clicks
                else:
                    pos_clicks = neg_clicks = 0

                rows.append({
                    "experiment_seed": self.seed,
                    "permutation_seed": ordering_index,
                    "image_index": index,
                    "iteration": iteration,
                    "commit_ground_truth": self.commit_ground_truth,
                    "dice_cutoff": self.dice_cutoff,
                    "pos_clicks": pos_clicks,
                    "neg_clicks": neg_clicks,
                    "score": float(score.item()),
                    "prompt_limit": self.prompt_iterations,
                })
                if score >= self.dice_cutoff:
                    break

            final_dice = float(score.item())
            iterations_used = iteration + 1
            image_summary_rows.append({
                "experiment_seed": self.seed,
                "permutation_seed": ordering_index,
                "image_index": index,
                "initial_dice": initial_dice,
                "final_dice": final_dice,
                "iterations_used": iterations_used,
                "reached_cutoff": final_dice >= self.dice_cutoff,
                "commit_type": "ground_truth" if self.commit_ground_truth else "prediction",
                "experiment_number": self.experiment_number,
                "protocol": self.interaction_protocol,
                "dice_cutoff": self.dice_cutoff,
                "prompt_limit": self.prompt_iterations
            })

            if update_context:
                binary_yhat = (yhat > 0).float()  # B x C x H x W
                mask_to_commit = (label[None, ...] if self.commit_ground_truth else binary_yhat)

                if context_images is None:
                    # Add a new "context axis" so final shape is (B, n, 1, H, W)
                    context_images = image[None, None, ...]        # (1, 1, 1, H, W)
                    context_labels = mask_to_commit[None, ...]  # (1, 1, 1, H, W)
                else:
                    # Append along the context dimension (dim=1)
                    context_images = torch.cat([context_images, image[None, None, ...]], dim=1)
                    context_labels = torch.cat([context_labels, mask_to_commit[None, ...]], dim=1)
        return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(image_summary_rows)
    
    def run_seq_multiverseg_eval(self, test_images, test_labels, ordering_index, seed_folder_dir):
        return self._run_seq_common(
            test_images,
            test_labels,
            ordering_index,
            seed_folder_dir,
            update_context=False,
        )

    def run_seq_multiverseg(self, support_images, support_labels, ordering_index, seed_folder_dir):
        return self._run_seq_common(
            support_images,
            support_labels,
            ordering_index,
            seed_folder_dir,
            update_context=True,
        )

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    train_split = 0.6
    d_support = WBCDataset('JTSC', split='support', label='nucleus', support_frac=train_split, testing_data_size=10)
    d_test = WBCDataset('JTSC', split='test', label='nucleus', support_frac=train_split, testing_data_size=10)
    with open(script_dir / "prompt_generator_configs/click_prompt_generator.yml", "r") as f:
        cfg = yaml.safe_load(f)
    prompt_generator_config = cfg['click_generator']
    prompt_generator =  eval_config(cfg)['click_generator']
    protocol_desc = (
        f"{prompt_generator_config.get('init_pos_click', 0)}_init_pos,"
        f"{prompt_generator_config.get('init_neg_click', 0)}_init_neg,"
        f"{prompt_generator_config.get('correction_clicks', 0)}_corrections"
    )
    experiment_number = 0
    experiment = MVSegOrderingExperiment(
        support_dataset=d_support, 
        test_dataset=d_test,
        prompt_generator=prompt_generator, 
        prompt_iterations=5, 
        commit_ground_truth=False, 
        permutations=1, 
        dice_cutoff=0.9, 
        interaction_protocol=f"{protocol_desc}",
        experiment_number=experiment_number,
        script_dir=script_dir
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
        
