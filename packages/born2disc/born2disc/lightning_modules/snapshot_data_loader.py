import torch as pt
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split, Subset

# DataModule: Organizes data preparation and loaders
class SnapshotDataLoaderModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset,
                 train_and_test_ratio=1.0,
                 train4train_ratio=0.75,
                 train4calibration_ratio=0.75, 
                 batch_size=512,
                 num_workers=31):
        super().__init__()
        self.dataset = dataset
        self.train_and_test_ratio = train_and_test_ratio
        self.train4train_ratio = train4train_ratio
        self.train4calibration_ratio = train4calibration_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_and_test_dataset = None
        self.calibration_dataset = None

        self.train_dataset = None
        self.test_dataset = None

        self.calibration_train_dataset = None
        self.calibration_test_dataset = None    

    def setup(self, stage=None):
        total_size = len(self.dataset)
        train_and_test_size = int(self.train_and_test_ratio * total_size)
        calibration_size = total_size - train_and_test_size

        # First split: main pool vs calibration pool (relative to original dataset)
        self.train_and_test_dataset, self.calibration_dataset = random_split(
            self.dataset, 
            [train_and_test_size, calibration_size]
        )

        # Keep first-level original indices
        train_and_test_indices_original = pt.tensor(self.train_and_test_dataset.indices, dtype=pt.long)
        calibration_indices_original    = pt.tensor(self.calibration_dataset.indices,    dtype=pt.long)

        # Second split: train vs test within the main pool (indices local to train_and_test_dataset)
        train4train_size = int(self.train4train_ratio * len(self.train_and_test_dataset))
        test4train_size  = len(self.train_and_test_dataset) - train4train_size

        train_dataset_local, test_dataset_local = random_split(
            self.train_and_test_dataset,
            [train4train_size, test4train_size]
        )

        # Resolve local -> original indices and rewrap as Subset(self.dataset, ...)
        train_dataset_indices_original = train_and_test_indices_original[pt.tensor(train_dataset_local.indices, dtype=pt.long)]
        test_dataset_indices_original  = train_and_test_indices_original[pt.tensor(test_dataset_local.indices,  dtype=pt.long)]

        self.train_dataset = Subset(self.dataset, train_dataset_indices_original.tolist())
        self.test_dataset  = Subset(self.dataset, test_dataset_indices_original.tolist())

        # Third split: calibration_train vs calibration_test within calibration pool
        train4calibration_size = int(self.train4calibration_ratio * len(self.calibration_dataset))
        test4calibration_size  = len(self.calibration_dataset) - train4calibration_size

        if len(self.calibration_dataset) > 0:
            calibration_train_local, calibration_test_local = random_split(
                self.calibration_dataset,
                [train4calibration_size, test4calibration_size]
            )

            calibration_train_indices_original = calibration_indices_original[pt.tensor(calibration_train_local.indices, dtype=pt.long)]
            calibration_test_indices_original  = calibration_indices_original[pt.tensor(calibration_test_local.indices,  dtype=pt.long)]

            self.calibration_train_dataset = Subset(self.dataset, calibration_train_indices_original.tolist())
            self.calibration_test_dataset  = Subset(self.dataset, calibration_test_indices_original.tolist())
        else:
            self.calibration_train_dataset = None
            self.calibration_test_dataset  = None


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)    
    
    def calibration_train_dataloader(self):
        if self.calibration_train_dataset is None:
            raise RuntimeError("calibration_train_dataset is None; set train_and_test_ratio < 1.0 to enable calibration pool.")
        return DataLoader(self.calibration_train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def calibration_val_dataloader(self):
        if self.calibration_test_dataset is None:
            raise RuntimeError("calibration_test_dataset is None; set train_and_test_ratio < 1.0 to enable calibration pool.")
        return DataLoader(self.calibration_test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)