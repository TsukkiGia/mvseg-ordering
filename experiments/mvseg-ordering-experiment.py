import os
os.environ['NEURITE_BACKEND'] = 'pytorch'

# add MultiverSeg, UniverSeg and ScribblePrompt dependencies
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dependencies = [PROJECT_ROOT / "MultiverSeg", PROJECT_ROOT / "UniverSeg", PROJECT_ROOT / "ScribblePrompt"]
for dependency in dependencies:
    if str(dependency) not in sys.path:
        sys.path.append(str(dependency))

script_dir = Path(__file__).resolve().parent  
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

import neurite as ne
from typing import Any
import torch
from dataset.wbc_multiple_perms import WBCDataset
from multiverseg.models.sp_mvs import MultiverSeg
import yaml
from pylot.experiment.util import eval_config
from score.dice_score import dice_score
import numpy as np
import matplotlib.pyplot as plt


class MVSegOrderingExperiment():
    def __init__(self, dataset: WBCDataset, 
                 prompt_generator: Any, 
                 prompt_iterations: int, 
                 commit_ground_truth: bool,
                 permutations: int,
                 seed: int = 23):
        self.dataset = dataset
        self.prompt_generator = prompt_generator
        self.model = MultiverSeg(version="v0")
        self.prompt_iterations = prompt_iterations
        self.commit_ground_truth = commit_ground_truth
        self.permutations = permutations
        self.seed = seed
        np.random.seed(seed)


    def run_permutations(self):
        base_indices = self.dataset.get_data_indices()
        for permutation_index in range(self.permutations):
            rng = np.random.default_rng(permutation_index)
            shuffled_indices = rng.permutation(base_indices).tolist()
            shuffled_data = [self.dataset[index] for index in shuffled_indices]
            support_images, support_labels = zip(*shuffled_data)
            support_images = torch.stack(support_images).to("cpu")
            support_labels = torch.stack(support_labels).to("cpu")
            self.run_seq_multiverseg(support_images, support_labels, permutation_index)
            break
    
    def run_seq_multiverseg(self, support_images, support_labels, ordering_index):
        # N x 1 x H x W for support images and labels
        assert(support_images.size(0) == support_labels.size(0))
        context_images = None
        context_labels = None
        for index in range(support_images.size(0)):
            
            # Image and Label: C x H x W
            image = support_images[index]
            label = support_labels[index]

            # For each image, we are doing max 20 iterations to get to 90 Dice
            for iteration in range(self.prompt_iterations):
                if iteration == 0:
                    prompts = self.prompt_generator(image[None], label[None])
                else:
                    prompts = self.prompt_generator.subsequent_prompt(
                            mask_pred=yhat, # shape: (1, C, H, W)
                            prev_input=prompts,
                            new_prompt=True
                    )
                
                annotations = {k:prompts.get(k) for k in ['point_coords', 'point_labels', 'mask_input', 'scribbles', 'box']}
                yhat = self.model.predict(image[None], context_images, context_labels, **annotations, return_logits=False).to('cpu')
                
                # visualize result
                fig, _ = ne.plot.slices([image.cpu(), label.cpu(), yhat > 0.5], width=10, 
                titles=['Image', 'Label', 'Prediction'])
                save_path = results_dir / f"Perm_{ordering_index}_image_{index}_prediction_{iteration}.png"
                fig.savefig(save_path)
                plt.close()
                score = dice_score(yhat > 0.5, label[None, ...])
                if score >= 0.9:
                    break

            # after all the iterations, update the context set
            binary_yhat = yhat[None, ...] > 0.5
            if context_images is None:
                # Image is C x H x W
                # We want 1 x n x C x H x W for context
                context_images = image[None, None, ...]
                # Label is C x H x W
                # yhat is n x C x H x W
                # We want 1 x n x C x H x W for context
                if self.commit_ground_truth:
                    context_labels = label[None, None, ...]
                else:
                    context_labels = binary_yhat
            else:
                context_images = torch.cat([context_images, image[None, None, ...]], dim=1)
                if self.commit_ground_truth:
                    context_labels = torch.cat([context_labels, label[None, None, ...]], dim=1)
                else:
                    context_labels = torch.cat([context_labels, binary_yhat], dim=1)
        

            

if __name__ == "__main__":
    train_split = 0.6
    d_support = WBCDataset('JTSC', split='support', label='nucleus', support_frac=train_split, testing_data_size=10)
    print("support",d_support.get_data_indices())
    d_test = WBCDataset('JTSC', split='test', label='nucleus', support_frac=train_split, testing_data_size=10)
    print("test", d_test.get_data_indices())
    with open(script_dir / "prompt_generator_configs/click_prompt_generator.yml", "r") as f:
        cfg = yaml.safe_load(f)  
    prompt_generator =  eval_config(cfg)['click_generator']
    print(prompt_generator)
    experiment = MVSegOrderingExperiment(d_support, prompt_generator, 5, False, 10)
    experiment.run_permutations()
        