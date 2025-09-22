from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from dataset.wbc_multiple_perms import WBCDataset

class MVSegOrderingExperiment():
    def __init__(self, dataset: WBCDataset, prompt_generator: Any):
        self.dataset = dataset
        self.prompt_generator = prompt_generator


    def run_permutations(self):
        for ordering_index in range(self.dataset.n_orderings):
            print(ordering_index)
            support_images, support_labels = self.dataset.get_data_from_ordering(ordering_index)
            support_images = torch.stack(support_images).to("cpu")
            support_labels = torch.stack(support_labels).to("cpu")

if __name__ == "__main__":
    train_split = 0.6
    d_support = WBCDataset('JTSC', split='support', label='nucleus', support_frac=train_split, testing_data_size=10)
    d_test = WBCDataset('JTSC', split='test', label='nucleus', support_frac=train_split, testing_data_size=10)  
    experiment = MVSegOrderingExperiment(d_support, None)
    experiment.run_permutations()
        