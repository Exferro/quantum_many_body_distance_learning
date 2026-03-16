import os

import torch as pt
import numpy as np

import pickle

from typing import Callable, Dict, List, Union
from dataclasses import dataclass, field


from nestconf import Config, Configurable

from ..dataset_classes.single_phase_diagram_point_dataset import SinglePhaseDiagramPointDataset
from ..dataset_classes.multiple_phase_diagram_points_dataset import MultiplePhaseDiagramPointsDataset
from ..lightning_modules.snapshot_data_loader import SnapshotDataLoaderModule
from ..lightning_modules.bootstrapped_snapshot_data_loader import BootstrappedSnapshotDataLoaderModule

from ..classification_backbones.feedforward import FeedforwardClassificationBackbone, FeedforwardClassificationBackboneConfig
from ..classification_backbones.convolutional import (
    ConvolutionalClassificationBackbone,
    ConvolutionalClassificationBackboneConfig,
)
from ..classification_backbones.pairwise_additive import (
    PairwiseAdditiveClassificationBackbone,
    PairwiseAdditiveClassificationBackboneConfig,
)

from ..lightning_modules.feedforward_classification import MetricsLoggerCallback, ClassificationLightningModule
from ..lightning_modules.calibration import TemperatureCalibrationLightningModule
from ..lightning_modules.f_divergence_callback import FDivergenceCallback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

@dataclass
class EarlyStoppingConfig(Config):
    monitor: str = 'val_loss'
    mode: str = 'min'
    patience: int = 3
    min_delta: float = 1e-4


class DistanceLearningPipeline(Configurable):
    phase_diagram_config: Config = None
    snapshots_per_point: int = 1000
    bootstrapped_snapshots_per_point: int = None
    take_every: int = 1
    bootstrap: bool = False
    bootstrap_size: int = None


    train_and_test_ratio: float = 1.0
    train4train_ratio: float = 0.75
    train4calibration_ratio: float = 0.75
    embedding_size: int = 64
    batch_size: int = 512

    classification_backbone_config: Config = None

    early_stopping_config: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    calibration_early_stopping_config: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    max_epochs: int = 100
    precision: int = 64
    dtype: str = "float64"
    learning_rate: float = 1e-3
    limit_train_batches: Union[int, float] = False
    limit_train_batches_val: Union[int, float] = None

    compute_transient_f_divergences: bool = False
    transient_f_divergence: str = 'hellinger'
    transient_selector: str = 'test'

    rng_seed: int = None

    def __init__(self,
                 *,
                 snapshots_root_dir: str = None,
                 save_root_dir: str = None, 
                 filter_lambda: Callable = None,
                 input_size_lambda: Callable = None,
                 bases_to_include: List[str] = None,
                 basis_prefix_dict: Dict[str, str] = None,
                 basis_label_dict: Dict[str, float] = None,
                 sanity_check_lambda: Callable = None,      
                 sorting_lambda: Callable = None,             
                 labels_generating_lambda: Callable = None,
                 use_deepset_encoding: bool = False,
                 accelerator: str = 'cpu',
                 **kwargs):
        super().__init__(**kwargs)

        self.dtype = self._resolve_dtype(self.dtype)

        if self.rng_seed is not None:
            pt.manual_seed(self.rng_seed)
            np.random.seed(self.rng_seed)

        assert snapshots_root_dir is not None        
        self.snapshots_root_dir = snapshots_root_dir

        assert save_root_dir is not None
        self.save_root_dir = save_root_dir


        self.self_dir = None
        # Calculate the directory for the current pipeline
        #assert self.phase_diagram_config is not None
        self.self_dir = os.path.join(self.save_root_dir, self.config.to_path_suffix())
        os.makedirs(self.self_dir, exist_ok=True)

        # Paths section
        self.path2model = os.path.join(self.self_dir, f'model.pth')
        
        self.filter_lambda = filter_lambda
        self.accelerator = accelerator
        
        self.input_size_lambda = input_size_lambda

        self.bases_to_include = bases_to_include
        self.basis_prefix_dict = basis_prefix_dict
        self.basis_label_dict = basis_label_dict

        self.sanity_check_lambda = sanity_check_lambda
        self.sorting_lambda = sorting_lambda

        self.use_deepset_encoding = use_deepset_encoding

        self.input_size, self.single_point_datasets = self.create_single_point_datasets()
        self.single_point_datasets = self.single_point_datasets[::self.take_every]
        print(len(self.single_point_datasets), 'single point datasets created')

        self.perms = []
        if self.snapshots_per_point is not None:
            for single_point_dataset in self.single_point_datasets:
                if single_point_dataset.perm is None:
                    raise NotImplementedError(f'It seems we have not updated yet correctly the snapshots_per_point for {self.__class__}')
                else:
                    self.perms.append(single_point_dataset.perm)
        self.num_points = len(self.single_point_datasets)

        self.labels_generating_lambda = labels_generating_lambda
        self.phase_diagram_dataset = MultiplePhaseDiagramPointsDataset(single_point_datasets=self.single_point_datasets,
                                                                       labels_generating_lambda=labels_generating_lambda,
                                                                       dtype=self.dtype)
        self.unbootstrapped_num_snapshots = len(self.phase_diagram_dataset)
        self.num_classes = self.phase_diagram_dataset.num_classes

        # Create Lightning data module
        self.data_module = SnapshotDataLoaderModule(dataset=self.phase_diagram_dataset,
                                                    train_and_test_ratio=self.train_and_test_ratio,
                                                    train4train_ratio=self.train4train_ratio,
                                                    train4calibration_ratio=self.train4calibration_ratio,
                                                    batch_size=self.batch_size,
                                                    num_workers=0)
        self.data_module.setup()

        if self.bootstrap:
            train_selector = pt.tensor(self.data_module.train_dataset.indices)
            test_selector = pt.tensor(self.data_module.test_dataset.indices)
            calibration_train_selector = pt.tensor(self.data_module.calibration_train_dataset.indices)
            calibration_test_selector = pt.tensor(self.data_module.calibration_test_dataset.indices)

            bootstrapped_train_selector = []
            bootstrapped_test_selector = []
            bootstrapped_calibration_train_selector = []
            bootstrapped_calibration_test_selector = []
            bootstrap_snapshots_counter = 0
            for dataset_idx, dataset in enumerate(self.single_point_datasets):
                train_pullback_indices = self.phase_diagram_dataset.global2local[train_selector[self.phase_diagram_dataset.point_labels[train_selector] == dataset_idx]]
                test_pullback_indices = self.phase_diagram_dataset.global2local[test_selector[self.phase_diagram_dataset.point_labels[test_selector] == dataset_idx]]
                calibration_train_pullback_indices = self.phase_diagram_dataset.global2local[calibration_train_selector[self.phase_diagram_dataset.point_labels[calibration_train_selector] == dataset_idx]]
                calibration_test_pullback_indices = self.phase_diagram_dataset.global2local[calibration_test_selector[self.phase_diagram_dataset.point_labels[calibration_test_selector] == dataset_idx]]
                bootstrap_train_indices, bootstrap_test_indices, bootstrap_calibration_train_indices, bootstrap_calibration_test_indices = dataset.bootstrap_train_test_new(train_indices=train_pullback_indices, 
                                             test_indices=test_pullback_indices,
                                             calibration_train_indices=calibration_train_pullback_indices,
                                             calibration_test_indices=calibration_test_pullback_indices,
                                             train_and_test_ratio=self.train_and_test_ratio,
                                             train4train_ratio=self.train4train_ratio,
                                             train4calibration_ratio=self.train4calibration_ratio)
                bootstrapped_train_selector.append(bootstrap_train_indices + bootstrap_snapshots_counter)
                bootstrapped_test_selector.append(bootstrap_test_indices + bootstrap_snapshots_counter)
                bootstrapped_calibration_train_selector.append(bootstrap_calibration_train_indices + bootstrap_snapshots_counter)
                bootstrapped_calibration_test_selector.append(bootstrap_calibration_test_indices + bootstrap_snapshots_counter)
                bootstrap_snapshots_counter += dataset.num_snapshots
            self.phase_diagram_dataset = MultiplePhaseDiagramPointsDataset(single_point_datasets=self.single_point_datasets,
                                                                           labels_generating_lambda=labels_generating_lambda,
                                                                           dtype=self.dtype)
            
            self.data_module = BootstrappedSnapshotDataLoaderModule(dataset=self.phase_diagram_dataset,
                                                                     bootstrapped_train_selector=pt.cat(bootstrapped_train_selector, dim=0),
                                                                     bootstrapped_test_selector=pt.cat(bootstrapped_test_selector, dim=0),
                                                                     bootstrapped_calibration_train_selector=pt.cat(bootstrapped_calibration_train_selector, dim=0),
                                                                     bootstrapped_calibration_test_selector=pt.cat(bootstrapped_calibration_test_selector, dim=0),
                                                                     batch_size=self.batch_size,
                                                                     num_workers=0)
            self.data_module.setup()
        # Create classification backbone: a model that takes a snapshot and returns an embedding
        self.classification_backbones = self.create_classification_backbone(input_size=self.input_size,
                                                            embedding_size=self.embedding_size,
                                                            num_classes=self.num_classes,
                                                        )
        self.classification_backbones = self.classification_backbones.to(self.dtype)

        # Create Lightning model which runs the classification task end-to-end
        self.model = ClassificationLightningModule(backbone=self.classification_backbones,
                                                num_classes=self.num_classes,
                                                learning_rate=self.learning_rate)  
        self.model = self.model.to(self.dtype)
        self.metrics_logger = None
        self.training_f_matrices = []
        self.calibration_f_matrices = []          
        if self.limit_train_batches:
            if self.limit_train_batches_val is None:
                cur_bootstrap_size = self.bootstrap_size if self.bootstrap else 1
                self.limit_train_batches_val = int(np.ceil((self.unbootstrapped_num_snapshots * self.train4calibration_ratio * self.train4train_ratio) / (cur_bootstrap_size * self.batch_size)))
        else:
            self.limit_train_batches_val = None

    @staticmethod
    def _resolve_dtype(dtype_value):
        if isinstance(dtype_value, pt.dtype):
            return dtype_value
        if isinstance(dtype_value, str):
            resolved = getattr(pt, dtype_value, None)
            if isinstance(resolved, pt.dtype):
                return resolved
        raise ValueError(f"Unsupported dtype value: {dtype_value}")

    def create_single_point_datasets(self):
        path_to_data_descr = os.path.join(self.snapshots_root_dir, 'data_descr.pickle')
        with open(path_to_data_descr, 'rb') as handle:
            data_descr = pickle.load(handle)

        
        single_point_datasets = []
        input_size = None
        for point_config in data_descr:
            if self.filter_lambda is not None:
                if not self.filter_lambda(point_config):
                    continue
            if input_size is None:
                input_size = self.input_size_lambda(point_config) + 1 # We always append a basis label
            else:
                assert (self.input_size_lambda(point_config) + 1) == input_size, f'Input size is not the same for all points: {input_size} != {self.input_size_lambda(point_config)}'
            cur_snapshots_dir = os.path.join(self.snapshots_root_dir,
                                             point_config.to_path_suffix())
            aggregated_snapshots = []
            for basis in self.bases_to_include:
                path_to_snapshot = os.path.join(cur_snapshots_dir,
                                                f'{self.basis_prefix_dict[basis]}snapshots.npy')
                if not os.path.exists(path_to_snapshot):
                    raise RuntimeError(f'{path_to_snapshot} does not exist for point {point_config}')
                snapshots = pt.from_numpy(np.load(path_to_snapshot)).type(self.dtype)
                sanity_check_result, sanity_check_message = self.sanity_check_lambda(point_config, snapshots)
                if not sanity_check_result:
                    raise RuntimeError(sanity_check_message)


                basis_label = self.basis_label_dict[basis]
                basis_labels = pt.full((snapshots.shape[0], 1), basis_label * 1.0, dtype=snapshots.dtype)
                basis_labelled_snapshots = pt.cat([basis_labels, snapshots], dim=1)

                assert basis_labelled_snapshots.shape[1] == input_size, f'basis_labelled_snapshots.shape[1] = {basis_labelled_snapshots.shape[1]} != input_size = {input_size}'
                aggregated_snapshots.append(basis_labelled_snapshots)   


            aggregated_snapshots = pt.cat(aggregated_snapshots, dim=0)
            single_point_datasets.append(SinglePhaseDiagramPointDataset(config=point_config,
                                                                        snapshots=aggregated_snapshots,
                                                                        sort=False,
                                                                        bootstrap=self.bootstrap,
                                                                        bootstrap_size=self.bootstrap_size,
                                                                        snapshots_per_point=self.snapshots_per_point,
                                                                        bootstrapped_snapshots_per_point=self.bootstrapped_snapshots_per_point,))
        single_point_datasets = self.sorting_lambda(single_point_datasets)
        
        return input_size * self.bootstrap_size if self.bootstrap else input_size, single_point_datasets

    def run_training(self,
            *,
            verbose_training: bool = False,
            enable_progress_bar: bool = True):
        early_stopping = EarlyStopping(**self.early_stopping_config.to_dict(), verbose=verbose_training)
        checkpoint_callback = ModelCheckpoint(dirpath=self.self_dir,
                                              filename="best",
                                              monitor=self.early_stopping_config.monitor,
                                              mode=self.early_stopping_config.mode,
                                              save_top_k=1,
                                              save_last=False,
                                              save_weights_only=True)

        self.metrics_logger = MetricsLoggerCallback()

        callbacks: List[pl.Callback] = [early_stopping, checkpoint_callback, self.metrics_logger]
        if self.compute_transient_f_divergences:
            f_divergence_callback = FDivergenceCallback(pipeline=self,
                                                        training_or_calibration='training')
            callbacks.append(f_divergence_callback)
        try:
            trainer = pl.Trainer(max_epochs=self.max_epochs,
                                accelerator=self.accelerator,
                                devices=1,
                                precision=self.precision,
                                log_every_n_steps=1,
                                callbacks=callbacks,
                                enable_progress_bar=enable_progress_bar,
                                enable_checkpointing=True,
                                limit_train_batches=self.limit_train_batches_val)
            trainer.fit(self.model, self.data_module)

        except pl.utilities.exceptions.MisconfigurationException:
            trainer = pl.Trainer(max_epochs=self.max_epochs,
                                accelerator=self.accelerator,
                                devices=1,
                                precision=self.precision,
                                log_every_n_steps=1,
                                callbacks=callbacks,
                                enable_progress_bar=enable_progress_bar,
                                enable_checkpointing=True,
                                limit_train_batches=1)
            trainer.fit(self.model, self.data_module)

        best_path = checkpoint_callback.best_model_path
        if best_path is None or best_path == "" or (not os.path.exists(best_path)):
            raise RuntimeError(
                "No best checkpoint was saved. "
                "Is val_loss being logged and is the monitor name correct?"
            )

        ckpt = pt.load(best_path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"], strict=True)

        # Optional: also save a simple state_dict where you already expect it
        pt.save(self.model.state_dict(), self.path2model)

        trainer._teardown()
            
    def create_classification_backbone(self, 
                                       *, 
                                       input_size: int = None,
                                       embedding_size: int = None,
                                       num_classes: int = None):
        if isinstance(self.classification_backbone_config, FeedforwardClassificationBackboneConfig):
            return FeedforwardClassificationBackbone(input_size=input_size,
                                                     embedding_size=embedding_size,
                                                     use_deepset_encoding=self.use_deepset_encoding,
                                                     bootstrap_size=self.bootstrap_size,
                                                     config=self.classification_backbone_config)
        elif isinstance(self.classification_backbone_config, ConvolutionalClassificationBackboneConfig):
            return ConvolutionalClassificationBackbone(input_size=input_size,
                                                       embedding_size=embedding_size,
                                                       config=self.classification_backbone_config)
        elif isinstance(self.classification_backbone_config, PairwiseAdditiveClassificationBackboneConfig):
            return PairwiseAdditiveClassificationBackbone(
                input_size=input_size,
                num_classes=num_classes,
                embedding_size=num_classes,
                config=self.classification_backbone_config,
            )
        else:
            raise ValueError(f'Unknown classification backbone config type: {type(self.classification_backbone_config)}')
        
    def calibrate_temperature_with_trainer(self,
                                       max_epochs_temperature: int = 100,
                                       learning_rate_temperature: float = 1e-3,
                                       verbose: bool = True):
        assert self.model is not None, "Train the model first."
        # Build the calibration wrapper
        calib_module = TemperatureCalibrationLightningModule(
            base_model=self.model,
            learning_rate_temperature=learning_rate_temperature
        )

        # Grab calibration loaders from your existing data_module
        calibration_train_loader = self.data_module.calibration_train_dataloader()
        calibration_val_loader = self.data_module.calibration_val_dataloader()

        early_stopping = EarlyStopping(**self.calibration_early_stopping_config.to_dict(),
                                    verbose=verbose)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.self_dir,
            filename="best_temp",
            monitor=self.calibration_early_stopping_config.monitor,
            mode=self.calibration_early_stopping_config.mode,
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        )
        

        callbacks: List[pl.Callback] = [early_stopping, checkpoint_callback]
        if self.compute_transient_f_divergences:
            f_divergence_callback = FDivergenceCallback(pipeline=self,
                                                        training_or_calibration='calibration')
            callbacks.append(f_divergence_callback)

        try:
            trainer = pl.Trainer(
                max_epochs=max_epochs_temperature,
                accelerator=self.accelerator,
                devices=1,
                precision=self.precision,
                enable_checkpointing=True,
                enable_progress_bar=verbose,
                callbacks=callbacks,
                log_every_n_steps=1,
                limit_train_batches=self.limit_train_batches_val        
            )
            # Fit only temperature
            trainer.fit(
                calib_module,
                train_dataloaders=calibration_train_loader,
                val_dataloaders=calibration_val_loader
            )
        except pl.utilities.exceptions.MisconfigurationException:
            trainer = pl.Trainer(max_epochs=self.max_epochs,
                    accelerator=self.accelerator,
                    devices=1,
                    precision=self.precision,
                    log_every_n_steps=1,
                    callbacks=callbacks,
                    enable_progress_bar=verbose,
                    enable_checkpointing=True,
                    limit_train_batches=1)
            # Fit only temperature
            trainer.fit(
                calib_module,
                train_dataloaders=calibration_train_loader,
                val_dataloaders=calibration_val_loader
            )

        # ---- restore best checkpoint (IMPORTANT) ----
        best_path = checkpoint_callback.best_model_path
        if best_path is None or best_path == "" or (not os.path.exists(best_path)):
            raise RuntimeError(
                "No best temperature checkpoint was saved. "
            )

        ckpt = pt.load(best_path, map_location="cpu")

        # Load into the calibration wrapper; this should restore the best temperature
        # (use strict=False just in case the wrapper stores extra keys)
        calib_module.load_state_dict(ckpt["state_dict"], strict=False)

        trainer._teardown()

        if verbose:
            T = self.model.get_temperature().item()
            print(f"[TempCalib] Final temperature = {T:.6f}")

    def calculate_f_divergences_matrix(self, 
                                *,
                                divergence: str = None,
                                selector: str = 'test',
                                use_importance_weights: bool = True,
                                balance_weights: bool = False,
                                importance_weight_selector: str = 'all',
                                symmetrize: bool = True):
        f_function = None
        match divergence:
            case 'hellinger':
                f_function = lambda density_ratio: (pt.sqrt(density_ratio) - 1)**2
            case 'triangular':
                f_function = lambda density_ratio: ((density_ratio - 1)**2 / (density_ratio + 1))
            case 'jensen_shannon':
                f_function = lambda density_ratio: 0.5 * (density_ratio * pt.log(density_ratio)
                                                    - (density_ratio + 1.0) * pt.log((density_ratio + 1.0) / 2.0))                
            case 'forward_kullback_leibler':
                f_function = lambda density_ratio: -pt.log(density_ratio) / pt.log(pt.tensor(2.0))
            case 'reverse_kullback_leibler':
                f_function = lambda density_ratio: density_ratio * pt.log(density_ratio) / pt.log(pt.tensor(2.0))
            case _:
                raise ValueError(f'Unknown divergence: {divergence}')
            
        match selector:
            case 'train':
                selector = pt.tensor(self.data_module.train_dataset.indices)
            case 'test':
                selector = pt.tensor(self.data_module.test_dataset.indices)
            case None:
                selector = pt.tensor(self.data_module.test_dataset.indices)
            case 'all':
                selector = pt.tensor(self.data_module.train_dataset.indices + self.data_module.test_dataset.indices)
            case _:
                raise ValueError(f'Unknown selector: {selector}')

        match importance_weight_selector:
            case 'train':
                importance_weight_selector = pt.tensor(self.data_module.train_dataset.indices)
            case 'test':
                importance_weight_selector = pt.tensor(self.data_module.test_dataset.indices)
            case None:
                importance_weight_selector = pt.tensor(self.data_module.test_dataset.indices)
            case 'all':
                importance_weight_selector = pt.tensor(self.data_module.train_dataset.indices + self.data_module.test_dataset.indices)
            case _:
                raise ValueError(f'Unknown importance_weight_selector: {importance_weight_selector}')    
        
        labels = self.phase_diagram_dataset.point_labels[selector]

        f_divergences_matrix = []
        f_divergences_matrix_std = []

        inverse_f_divergences_matrix = []
        inverse_f_divergences_matrix_std = []

        inverse_dataset_labels = []
        per_class_batch_sizes = []
        all_labels = self.phase_diagram_dataset.point_labels[importance_weight_selector]
        for dataset_idx, dataset in enumerate(self.single_point_datasets):
            inverse_dataset_labels.append(self.phase_diagram_dataset.labels[selector[labels == dataset_idx]][0].item())
            per_class_batch_sizes.append((all_labels == dataset_idx).sum().item())
        inverse_dataset_labels = pt.tensor(inverse_dataset_labels)
        per_class_batch_sizes = pt.tensor(per_class_batch_sizes, dtype=self.dtype)
        
        for dataset_idx, dataset in enumerate(self.single_point_datasets):
            pullback_indices = self.phase_diagram_dataset.global2local[selector[labels == dataset_idx]]
            cur_data = dataset.snapshots[pullback_indices]
            assert cur_data.shape[0] > 0, f'No data for dataset {dataset_idx} with selector {selector}'
            self.model.eval()
            with pt.no_grad():
                logits, _ = self.model(cur_data)
            class_densities = pt.softmax(logits, dim=-1)
            class_densities = class_densities[:, inverse_dataset_labels]
            class_densities = class_densities / pt.sum(class_densities, dim=-1, keepdims=True)
                
            posterior_density_ratios = class_densities / class_densities[:, dataset_idx:dataset_idx+1]
            if use_importance_weights:
                if balance_weights:
                    prior_density_ratios = (per_class_batch_sizes[dataset_idx] / per_class_batch_sizes) * posterior_density_ratios
                    class_priors = (per_class_batch_sizes[dataset_idx] / (per_class_batch_sizes + per_class_batch_sizes[dataset_idx]))
                else:
                    prior_density_ratios = posterior_density_ratios
                    class_priors = 0.5
                importance_weights = 1. / (class_priors + (1. - class_priors) * prior_density_ratios)
            else:
                prior_density_ratios = posterior_density_ratios
                importance_weights = pt.ones_like(posterior_density_ratios)
                class_priors = 1.0

            per_snapshot_f_divergences = importance_weights * f_function(prior_density_ratios)

            f_divergences_matrix.append(per_snapshot_f_divergences.mean(dim=0) * class_priors)
            f_divergences_matrix_std.append(per_snapshot_f_divergences.std(dim=0) * class_priors/ pt.sqrt(pt.tensor(per_snapshot_f_divergences.shape[0] * 1.0)))

            inverse_posterior_density_ratios = (class_densities[:, dataset_idx:dataset_idx+1] / class_densities)
            if use_importance_weights:
                if balance_weights:
                    inverse_prior_density_ratios = (per_class_batch_sizes / per_class_batch_sizes[dataset_idx]) * inverse_posterior_density_ratios
                    inverse_class_priors = (per_class_batch_sizes / (per_class_batch_sizes + per_class_batch_sizes[dataset_idx]))
                else:   
                    inverse_prior_density_ratios = inverse_posterior_density_ratios
                    inverse_class_priors = 0.5                
                inverse_importance_weights = 1. / (inverse_class_priors + (1. - inverse_class_priors) * inverse_prior_density_ratios)
            else:
                inverse_prior_density_ratios = inverse_posterior_density_ratios
                inverse_importance_weights = pt.ones_like(inverse_posterior_density_ratios)
                inverse_class_priors = 1.0
 
            inverse_per_snapshot_f_divergences = inverse_importance_weights * f_function(inverse_prior_density_ratios)

            inverse_f_divergences_matrix.append(inverse_per_snapshot_f_divergences.mean(dim=0) * (1. - inverse_class_priors))
            inverse_f_divergences_matrix_std.append(inverse_per_snapshot_f_divergences.std(dim=0) * (1. - inverse_class_priors) / pt.sqrt(pt.tensor(inverse_per_snapshot_f_divergences.shape[0] * 1.0)))
            
        f_divergences_matrix = pt.stack(f_divergences_matrix, dim=0)
        f_divergences_matrix_std = pt.stack(f_divergences_matrix_std, dim=0)

        inverse_f_divergences_matrix = pt.stack(inverse_f_divergences_matrix, dim=0)
        inverse_f_divergences_matrix_std = pt.stack(inverse_f_divergences_matrix_std, dim=0)
        
        if use_importance_weights:
            f_divergences_matrix = f_divergences_matrix + inverse_f_divergences_matrix.T
            f_divergences_matrix_std = 0.5 * pt.sqrt(f_divergences_matrix_std**2 + inverse_f_divergences_matrix_std.T**2)

        if symmetrize:
            f_divergences_matrix = 0.5 * (f_divergences_matrix + f_divergences_matrix.T)
            f_divergences_matrix_std = 0.5 * (f_divergences_matrix_std + f_divergences_matrix_std.T)

        return f_divergences_matrix, f_divergences_matrix_std
    
    def calculate_fisher_information_odd_stencil_delta_multiplier(
        self,
        *,
        selector: str = 'test',
        stencil_size: int = 3,
        delta_multiplier: int = 1,
        center_score_proxy: bool = True,
    ):
        """
        Estimates Fisher information at interior points using a symmetric odd-point central stencil
        on logits, with optional index spacing controlled by delta_multiplier.

        We approximate the score (first derivative) of log p_theta via logits z_k(x) as:
            score(x) ≈ (1 / Δθ) * Σ_j w_j * z_{k + j * delta_multiplier}(x)

        This function RETURNS the Δθ-free quantity:
            score_proxy(x) = Σ_j w_j * z_{k + j * delta_multiplier}(x)

        where the weights w_j include ALL stencil numerical constants and also include the
        dimensionless 1/delta_multiplier factor (since effective step is delta_multiplier * Δθ),
        but DO NOT include division by the dimensional Δθ.

        For stencil_size = 2m+1, valid k are:
            k = m*delta_multiplier .. (N-1 - m*delta_multiplier)
        so you get N - 2*m*delta_multiplier outputs.

        Returns:
            fisher_information: Tensor [N - 2*m*delta_multiplier]
            fisher_information_std: Tensor [N - 2*m*delta_multiplier]
            used_dataset_indices: List[int]  (the k values where Fisher was evaluated)
        """
        def _finite_difference_weights_central_first_derivative(
            *,
            half_width_m: int,
            delta_multiplier: int,
            dtype: pt.dtype,
            device: pt.device,
        ) -> pt.Tensor:
            """
            Fornberg algorithm: weights for the first derivative at x0=0,
            using grid points x = (-m*d), ..., 0, ..., (+m*d) with d=delta_multiplier.

            Then:
                f'(0) ≈ Σ_j w_j f(x_j)
            and if x_j are in "index units", converting to θ uses:
                ∂θ f ≈ (1/Δθ) Σ_j w_j f(k + j*d)

            So w_j already includes the 1/d factor (dimensionless), but not 1/Δθ.
            """
            m = half_width_m
            d = delta_multiplier
            if d < 1:
                raise ValueError(f"delta_multiplier must be >= 1, got {d}.")

            x = (pt.arange(-m, m + 1, dtype=dtype, device=device) * pt.tensor(float(d), dtype=dtype, device=device))
            x0 = pt.tensor(0.0, dtype=dtype, device=device)

            n = x.numel()
            # Need weights up to first derivative only
            c = pt.zeros((n, 2), dtype=dtype, device=device)
            c1 = pt.tensor(1.0, dtype=dtype, device=device)
            c4 = x[0] - x0
            c[0, 0] = 1.0

            for i in range(1, n):
                mn = min(i, 1)
                c2 = pt.tensor(1.0, dtype=dtype, device=device)
                c5 = c4
                c4 = x[i] - x0
                for j in range(0, i):
                    c3 = x[i] - x[j]
                    c2 = c2 * c3
                    if j == i - 1:
                        c[i, 0] = (-c1 * c5 / c2) * c[i - 1, 0]
                        c[i, 1] = (c1 / c2) * (c[i - 1, 0] - c5 * c[i - 1, 1])
                    c[j, 1] = (c4 * c[j, 1] - c[j, 0]) / c3
                    c[j, 0] = (c4 * c[j, 0]) / c3
                c1 = c2

            return c[:, 1]  # first-derivative weights

        if (stencil_size < 3) or (stencil_size % 2 != 1):
            raise ValueError(f"stencil_size must be an odd integer >= 3, got {stencil_size}.")
        if delta_multiplier < 1:
            raise ValueError(f"delta_multiplier must be >= 1, got {delta_multiplier}.")

        match selector:
            case 'train':
                selector_tensor = pt.tensor(self.data_module.train_dataset.indices)
            case 'test':
                selector_tensor = pt.tensor(self.data_module.test_dataset.indices)
            case None:
                selector_tensor = pt.tensor(self.data_module.test_dataset.indices)
            case 'all':
                selector_tensor = pt.tensor(
                    self.data_module.train_dataset.indices
                    + self.data_module.test_dataset.indices
                )
            case _:
                raise ValueError(f'Unknown selector: {selector}')

        labels = self.phase_diagram_dataset.point_labels[selector_tensor]

        num_points = len(self.single_point_datasets)
        m = (stencil_size - 1) // 2
        half_span = m * delta_multiplier

        if num_points < (2 * half_span + 1):
            raise ValueError(
                f'Need at least {2 * half_span + 1} datasets for stencil_size={stencil_size} '
                f'and delta_multiplier={delta_multiplier}, got {num_points}.'
            )

        weights = _finite_difference_weights_central_first_derivative(
            half_width_m=m,
            delta_multiplier=delta_multiplier,
            dtype=self.dtype,
            device=pt.device('cpu'),
        )  # [2m+1]

        fisher_information = []
        fisher_information_std = []
        used_dataset_indices: List[int] = []

        self.model.eval()
        with pt.no_grad():
            for k in range(half_span, num_points - half_span):
                dataset_k = self.single_point_datasets[k]

                cur_selector = selector_tensor[labels == k]
                pullback_indices = self.phase_diagram_dataset.global2local[cur_selector]

                cur_data = dataset_k.snapshots[pullback_indices]
                if cur_data.shape[0] == 0:
                    raise RuntimeError(
                        f'No data for dataset index {k} with selector={selector}.'
                    )

                logits, _ = self.model(cur_data)  # [B, N]

                # Collect logits at k + j*delta_multiplier for j=-m..m
                col_indices = [k + j * delta_multiplier for j in range(-m, m + 1)]
                cols = logits[:, col_indices]  # [B, 2m+1]

                # score_proxy(x) = Σ_j w_j * z_{k + j*d}(x)   (no division by Δθ)
                score_proxy = cols @ weights  # [B]

                if center_score_proxy:
                    score_proxy = score_proxy - score_proxy.mean()

                per_sample_contrib = score_proxy ** 2
                I_hat = per_sample_contrib.mean()
                I_hat_se = (
                    per_sample_contrib.std(unbiased=False)
                    / pt.sqrt(pt.tensor(per_sample_contrib.shape[0], dtype=per_sample_contrib.dtype))
                )

                fisher_information.append(I_hat)
                fisher_information_std.append(I_hat_se)
                used_dataset_indices.append(k)

        fisher_information = pt.stack(fisher_information, dim=0)
        fisher_information_std = pt.stack(fisher_information_std, dim=0)

        return fisher_information, fisher_information_std, used_dataset_indices
