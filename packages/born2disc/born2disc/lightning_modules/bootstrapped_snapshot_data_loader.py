import torch as pt
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset

# DataModule: Organizes data preparation and loaders
class BootstrappedSnapshotDataLoaderModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset,
                 bootstrapped_train_selector: pt.Tensor = None,
                 bootstrapped_test_selector: pt.Tensor = None,
                 bootstrapped_calibration_train_selector: pt.Tensor = None,
                 bootstrapped_calibration_test_selector: pt.Tensor = None,
                 batch_size=512,
                 num_workers=31):
        super().__init__()
        self.dataset = dataset

        self.bootstrapped_train_selector = bootstrapped_train_selector
        self.bootstrapped_test_selector = bootstrapped_test_selector
        self.bootstrapped_calibration_train_selector = bootstrapped_calibration_train_selector
        self.bootstrapped_calibration_test_selector = bootstrapped_calibration_test_selector

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_and_test_dataset = None
        self.calibration_dataset = None

        self.train_dataset = None
        self.test_dataset = None

        self.calibration_train_dataset = None
        self.calibration_test_dataset = None    

    def setup(self, stage=None):
        self.train_and_test_dataset = Subset(self.dataset, self.bootstrapped_train_selector.tolist() + self.bootstrapped_test_selector.tolist())
        self.calibration_dataset    = Subset(self.dataset, self.bootstrapped_calibration_train_selector.tolist() + self.bootstrapped_calibration_test_selector.tolist())

        self.train_dataset = Subset(self.dataset, self.bootstrapped_train_selector.tolist())
        self.test_dataset  = Subset(self.dataset, self.bootstrapped_test_selector.tolist())

        self.calibration_train_dataset = Subset(self.dataset, self.bootstrapped_calibration_train_selector.tolist())
        self.calibration_test_dataset  = Subset(self.dataset, self.bootstrapped_calibration_test_selector.tolist())

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
