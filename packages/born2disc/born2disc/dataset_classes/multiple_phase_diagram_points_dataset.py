from typing import List, Callable

import numpy as np
import torch as pt
from torch.utils.data import Dataset, ConcatDataset

from ..dataset_classes.single_phase_diagram_point_dataset import SinglePhaseDiagramPointDataset


# Custom dataset class for SimCLR contrastive learning
class MultiplePhaseDiagramPointsDataset(Dataset):
    """
    A dataset class for SimCLR contrastive learning.
    """
    def __init__(self,
                 *,
                 single_point_datasets: List[SinglePhaseDiagramPointDataset] = None,
                 labels_generating_lambda: Callable = None,
                 dtype: pt.dtype = pt.float64):
        super().__init__()

        self.single_point_datasets = single_point_datasets
        self.num_single_points = len(single_point_datasets)

        # Prepare the data for all parameters
        #self.data = pt.cat([single_point_dataset.snapshots for single_point_dataset in self.single_point_datasets], dim=0)
        self.concat_dataset = ConcatDataset(single_point_datasets)

        self.labels_generating_lambda = labels_generating_lambda
        self.dtype = dtype
        if labels_generating_lambda is not None:
            self.labels = labels_generating_lambda([single_point_dataset for single_point_dataset in single_point_datasets])
        else:   
            self.labels = pt.cat([pt.full((len(single_point_dataset),), idx, dtype=pt.long) for idx, single_point_dataset in enumerate(self.single_point_datasets)], dim=0)
        self.point_labels = pt.cat([pt.full((len(single_point_dataset),), idx, dtype=pt.long) for idx, single_point_dataset in enumerate(self.single_point_datasets)], dim=0)
        self.num_classes = len(np.unique(self.labels.numpy()))

        self.single_point_datasets_sizes = pt.tensor([len(single_point_dataset) for single_point_dataset in self.single_point_datasets])
        self.single_point_datasets_offsets = pt.cumsum(self.single_point_datasets_sizes, dim=0) - self.single_point_datasets_sizes[0]
        self.global2local = pt.cat([pt.arange(size) for size in self.single_point_datasets_sizes], dim=0)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        """
        Returns a random snapshot, ignoring the labels.

        Args:
            idx: Index (not used, snapshots are randomly sampled).

        Returns:
            A random snapshot.
        """
        snapshot = self.concat_dataset[idx]
        label = pt.nn.functional.one_hot(self.labels[idx], num_classes=self.num_single_points).type(self.dtype)
        return snapshot, label
