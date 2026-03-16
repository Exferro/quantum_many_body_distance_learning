import torch as pt
from torch.utils.data import Dataset

from nestconf import Config

from ..utils.digit_vector_processor import DigitVectorProcessor


# Custom dataset class to handle snapshot images for a single parameter
class SinglePhaseDiagramPointDataset(Dataset):
    """
    A class to create a dataset for a single parameter storing all snapshots in a concatenated way.
    """
    def __init__(
        self,
        *,
        config: Config = None,
        snapshots: pt.Tensor = None,
        digit_vec_proc: DigitVectorProcessor = None,
        bootstrap: bool = False,
        bootstrap_size: int = None,
        sort: bool = False,
        snapshots_per_point: int = None,
        bootstrapped_snapshots_per_point: int = None,
    ):
        assert config is not None, "Config must be provided."
        self.config = config
        assert snapshots.ndim == 2, "Snapshots should be a 2D tensor."

        self.digit_vec_proc = None

        if sort:
            if digit_vec_proc is None:
                self.digit_vec_proc = DigitVectorProcessor(radix=len(pt.unique(snapshots)),
                                                        digits_num=snapshots.shape[1])
                #pass
            else:
                self.digit_vec_proc = digit_vec_proc
            unique_pixel_values = pt.unique(snapshots).type(pt.int64)
            if not (pt.all((-(self.digit_vec_proc.radix - 1) // 2) <= unique_pixel_values) 
                    and pt.all(unique_pixel_values <= self.digit_vec_proc.radix // 2)):
                raise ValueError(f"Unique pixel values are not in the correct range to sort them with the digit vector processor."
                                 f"Unique pixel values are {unique_pixel_values}, while the required range is "
                                 f"{(-(self.digit_vec_proc.radix - 1) // 2, self.digit_vec_proc.radix // 2)} inclusive.")
            self.snapshots, _ = self.digit_vec_proc.sort_digit_vector(digit_vector=snapshots)
        else:
            self.snapshots = snapshots
        self.num_snapshots = self.snapshots.shape[0] 

        if bootstrap:
            assert bootstrap_size is not None, "Bootstrap size must be provided if bootstrap is True."
            self.bootstrap_size = bootstrap_size
            # self.bootstrap_indices = pt.randint(0, self.num_snapshots, (self.num_snapshots, bootstrap_size,))
            # bootstrap_snapshots = self.snapshots[self.bootstrap_indices]
            #self.snapshots = bootstrap_snapshots.reshape(self.num_snapshots, -1)
            

        if (bootstrap is False) and (bootstrap_size is not None):
            raise ValueError("Bootstrap flag must be set to True to enable bootstrap size.")
        
        self.snapshots_per_point = snapshots_per_point
        self.perm = None
        if self.snapshots_per_point is not None:
            if self.snapshots.shape[0] > self.snapshots_per_point:
                self.perm  = pt.randperm(self.snapshots.shape[0])[:self.snapshots_per_point]
                self.snapshots = self.snapshots[self.perm]
                self.num_snapshots = self.snapshots.shape[0]
            else:
                self.perm = pt.arange(self.snapshots.shape[0])
                self.snapshots = self.snapshots
                self.num_snapshots = self.snapshots.shape[0]
        self.bootstrap = bootstrap
        if self.bootstrap:
            self.original_snapshots = self.snapshots.clone()
        else:
            self.original_snapshots = self.snapshots
        self.bootstrapped_snapshots_per_point = bootstrapped_snapshots_per_point

    def __len__(self):
        return self.num_snapshots

    def __getitem__(self, idx):
        return self.snapshots[idx]
    
    def bootstrap_train_test(self, 
                             train_indices: pt.Tensor, 
                             test_indices: pt.Tensor,
                             calibration_train_indices: pt.Tensor = None,
                             calibration_test_indices: pt.Tensor = None,):
        """
        Returns the bootstrap train and test datasets.
        
        Args:
            train_indices: Indices for the training set.
            test_indices: Indices for the test set.
            calibration_train_indices: Indices for the calibration training set.
            calibration_test_indices: Indices for the calibration test set.
        
        Returns:
            A tuple of (train_dataset, test_dataset).
        """
        num_train_snapshots = train_indices.shape[0]
        num_test_snapshots = test_indices.shape[0]
        num_calibration_train_snapshots = calibration_train_indices.shape[0]
        num_calibration_test_snapshots = calibration_test_indices.shape[0]
        assert num_train_snapshots + num_test_snapshots + num_calibration_train_snapshots + num_calibration_test_snapshots == self.num_snapshots
        
        train_snapshots = self.snapshots[train_indices]
        train_bootstrap_indices = pt.randint(0, num_train_snapshots, (num_train_snapshots, self.bootstrap_size,))
        train_bootstrap_snapshots = train_snapshots[train_bootstrap_indices].reshape(num_train_snapshots, -1)

        test_snapshots = self.snapshots[test_indices]
        test_bootstrap_indices = pt.randint(0, num_test_snapshots, (num_test_snapshots, self.bootstrap_size,))
        test_bootstrap_snapshots = test_snapshots[test_bootstrap_indices].reshape(num_test_snapshots, -1)

        calibration_train_snaphots = self.snapshots[calibration_train_indices]
        calibration_train_bootstrap_indices = pt.randint(0, num_calibration_train_snapshots, (num_calibration_train_snapshots, self.bootstrap_size,))
        calibration_train_bootstrap_snapshots = calibration_train_snaphots[calibration_train_bootstrap_indices].reshape(num_calibration_train_snapshots, -1)

        calibration_test_snaphots = self.snapshots[calibration_test_indices]
        calibration_test_bootstrap_indices = pt.randint(0, num_calibration_test_snapshots, (num_calibration_test_snapshots, self.bootstrap_size,))
        calibration_test_bootstrap_snapshots = calibration_test_snaphots[calibration_test_bootstrap_indices].reshape(num_calibration_test_snapshots, -1)

        bootstrap_snapshots = pt.cat((train_bootstrap_snapshots,
                                      test_bootstrap_snapshots, 
                                      calibration_train_bootstrap_snapshots, 
                                      calibration_test_bootstrap_snapshots), dim=0)
        bootstrap_snapshots = pt.zeros_like(bootstrap_snapshots)
        bootstrap_snapshots[train_indices] = train_bootstrap_snapshots
        bootstrap_snapshots[test_indices] = test_bootstrap_snapshots
        bootstrap_snapshots[calibration_train_indices] = calibration_train_bootstrap_snapshots
        bootstrap_snapshots[calibration_test_indices] = calibration_test_bootstrap_snapshots

        self.snapshots = bootstrap_snapshots

        return None
    
    def bootstrap_train_test_new(self, 
                             train_indices: pt.Tensor, 
                             test_indices: pt.Tensor,
                             calibration_train_indices: pt.Tensor = None,
                             calibration_test_indices: pt.Tensor = None,
                             train_and_test_ratio: float = 1.0,
                             train4train_ratio: float = 0.75,
                             train4calibration_ratio: float = 0.75):
        """
        Returns the bootstrap train and test datasets.
        
        Args:
            train_indices: Indices for the training set.
            test_indices: Indices for the test set.
            calibration_train_indices: Indices for the calibration training set.
            calibration_test_indices: Indices for the calibration test set.
        
        Returns:
            A tuple of (train_dataset, test_dataset).
            
        """
        assert (train_indices.shape[0] 
                + test_indices.shape[0] 
                + calibration_train_indices.shape[0] 
                + calibration_test_indices.shape[0]) == self.num_snapshots
        total_size = self.bootstrapped_snapshots_per_point
        train_and_test_size = int(train_and_test_ratio * total_size)
        calibration_size = total_size - train_and_test_size

        train4train_size = int(train4train_ratio * train_and_test_size)
        test4train_size  = train_and_test_size - train4train_size
        train4calibration_size = int(train4calibration_ratio * calibration_size)
        test4calibration_size  = calibration_size - train4calibration_size

        train_snapshots = self.snapshots[train_indices]
        train_bootstrap_indices = pt.randint(0, train_indices.shape[0], (train4train_size, self.bootstrap_size,))
        train_bootstrap_snapshots = train_snapshots[train_bootstrap_indices].reshape(train4train_size, -1)

        test_snapshots = self.snapshots[test_indices]
        test_bootstrap_indices = pt.randint(0, test_indices.shape[0], (test4train_size, self.bootstrap_size,))
        test_bootstrap_snapshots = test_snapshots[test_bootstrap_indices].reshape(test4train_size, -1)

        calibration_train_snaphots = self.snapshots[calibration_train_indices]
        calibration_train_bootstrap_indices = pt.randint(0, calibration_train_indices.shape[0], (train4calibration_size, self.bootstrap_size,))
        calibration_train_bootstrap_snapshots = calibration_train_snaphots[calibration_train_bootstrap_indices].reshape(train4calibration_size, -1) 

        calibration_test_snaphots = self.snapshots[calibration_test_indices]
        calibration_test_bootstrap_indices = pt.randint(0, calibration_test_indices.shape[0], (test4calibration_size, self.bootstrap_size,))
        calibration_test_bootstrap_snapshots = calibration_test_snaphots[calibration_test_bootstrap_indices].reshape(test4calibration_size, -1)

        bootstrap_snapshots = pt.cat((train_bootstrap_snapshots,
                                      test_bootstrap_snapshots, 
                                      calibration_train_bootstrap_snapshots, 
                                      calibration_test_bootstrap_snapshots), dim=0)
        
        self.snapshots = bootstrap_snapshots
        self.num_snapshots = self.snapshots.shape[0]

        bootstrap_train_indices = pt.arange(0, train4train_size)
        bootstrap_test_indices = pt.arange(train4train_size, train4train_size + test4train_size)
        bootstrap_calibration_train_indices = pt.arange(train4train_size + test4train_size, 
                                                       train4train_size + test4train_size + train4calibration_size)
        bootstrap_calibration_test_indices = pt.arange(train4train_size + test4train_size + train4calibration_size, 
                                                      total_size)

        return (bootstrap_train_indices, 
                bootstrap_test_indices,
                bootstrap_calibration_train_indices,
                bootstrap_calibration_test_indices)

