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


class MVSegOrderingExperiment():
    def __init__(self, dataset: WBCDataset, prompt_generator: Any, prompt_iterations: int, commit_ground_truth: bool):
        self.dataset = dataset
        self.prompt_generator = prompt_generator
        self.model = MultiverSeg(version="v0")
        self.prompt_iterations = prompt_iterations
        self.commit_ground_truth = commit_ground_truth


    def run_permutations(self):
        for ordering_index in range(len(self.dataset.orderings)):
            support_images, support_labels = self.dataset.get_data_from_ordering(ordering_index)
            support_images = torch.stack(support_images).to("cpu")
            support_labels = torch.stack(support_labels).to("cpu")
            self.run_seq_multiverseg(support_images, support_labels, ordering_index)
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
                titles=['Image', 'Label', 'MultiverSeg'])
                save_path = results_dir / f"{ordering_index}_prediction_{iteration}.png"
                fig.savefig(save_path)
                score = dice_score(yhat > 0.5, label[None, ...])
                if score >= 0.9:
                    break
            
            break
        

            

if __name__ == "__main__":
    train_split = 0.6
    d_support = WBCDataset('JTSC', split='support', label='nucleus', support_frac=train_split, testing_data_size=10)
    d_test = WBCDataset('JTSC', split='test', label='nucleus', support_frac=train_split, testing_data_size=10)
    with open(script_dir / "prompt_generator_configs/click_prompt_generator.yml", "r") as f:
        cfg = yaml.safe_load(f)  
    prompt_generator =  eval_config(cfg)['click_generator']
    print(prompt_generator)
    experiment = MVSegOrderingExperiment(d_support, prompt_generator, 20, False)
    experiment.run_permutations()
        