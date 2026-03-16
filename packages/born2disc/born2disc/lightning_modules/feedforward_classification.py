import torch as pt
import pytorch_lightning as pl

import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class ClassificationBackbone(pt.nn.Module, ABC):
    call_super_init = True

    def __init__(self, *args, embedding_size: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert embedding_size is not None, "embedding_size must be provided"
        self.embedding_size = embedding_size

    @abstractmethod
    def forward(self, x: pt.Tensor, dtype: pt.dtype = None) -> pt.Tensor:
        pass


class MetricsLoggerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in metrics:
            if 'train' in key:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(metrics[key].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in metrics:
            if 'val' in key:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(metrics[key].item())

    def to_pandas(self):
        df = pd.DataFrame({k: pd.Series(v) for k, v in self.metrics.items()})
        df.index.name = "epoch"
        df = df.reset_index()

        preferred = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        return df[cols]


class ClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        backbone: ClassificationBackbone,
        num_classes: int,
        learning_rate: float = 0.001,
        *,
        symmetry_pooling: bool = False,
        symmetry_use_d4: bool = False,
        symmetry_use_spinflip: bool = False,
        symmetry_pool_embeddings: bool = False,
        symmetry_eps: float = 1e-12,
    ):
        super().__init__()
        print(f'We use symmetry pooling: {symmetry_pooling}')
        self.save_hyperparameters(ignore=["backbone"])

        if not isinstance(backbone, ClassificationBackbone):
            raise TypeError("backbone must be a ClassificationBackbone")
        self.backbone = backbone

        self.backbone_produces_logits = bool(getattr(self.backbone, "produces_logits", False))

        if self.backbone_produces_logits:
            if int(self.backbone.embedding_size) != int(num_classes):
                raise ValueError(
                    f"Backbone produces logits, so backbone.embedding_size must equal num_classes. "
                    f"Got embedding_size={self.backbone.embedding_size}, num_classes={num_classes}."
                )
            self.classification_head = pt.nn.Identity()
        else:
            self.classification_head = pt.nn.Linear(
                self.backbone.embedding_size,
                num_classes,
            )


        self.criterion = pt.nn.CrossEntropyLoss()

        self.log_temperature = pt.nn.Parameter(pt.zeros(1), requires_grad=False)

        # --- Symmetry pooling config ---
        self.symmetry_pooling = symmetry_pooling
        self.symmetry_use_d4 = symmetry_use_d4
        self.symmetry_use_spinflip = symmetry_use_spinflip
        self.symmetry_pool_embeddings = symmetry_pool_embeddings
        self.symmetry_eps = symmetry_eps

        if self.symmetry_pooling and (not bool(getattr(self.backbone, "supports_symmetry_pooling", True))):
            self.symmetry_pooling = False

    def get_temperature(self) -> pt.Tensor:
        temperature = pt.exp(self.log_temperature)
        return temperature.clamp(min=1e-3, max=1e3)

    def enable_temperature_calibration(self):
        self.log_temperature.requires_grad_(True)

    def disable_temperature_calibration(self):
        self.log_temperature.requires_grad_(False)

    # -----------------------
    # Symmetry helpers
    # -----------------------
    def _split_basis_and_image_flat(
        self,
        *,
        x_flat: pt.Tensor,
    ) -> Tuple[Optional[pt.Tensor], pt.Tensor]:
        basis_label_included = bool(getattr(self.backbone, "basis_label_included", False))
        basis_label_size = int(getattr(self.backbone, "basis_label_size", 0))

        if basis_label_included:
            basis = x_flat[:, :basis_label_size]
            img_flat = x_flat[:, basis_label_size:]
            return basis, img_flat

        return None, x_flat

    def _infer_image_shape(self) -> Tuple[int, int, int]:
        # Expect backbone to have these (your backbone does).
        input_channels = int(getattr(self.backbone, "input_channels", 1))
        input_shape = getattr(self.backbone, "input_shape", None)
        if input_shape is None:
            raise ValueError("Backbone must expose input_shape=(H,W) for symmetry pooling.")
        h, w = int(input_shape[0]), int(input_shape[1])
        if h != w:
            raise ValueError(f"Symmetry pooling expects square lattice; got input_shape={input_shape}.")
        return input_channels, h, w

    def _flat_to_image(
        self,
        *,
        img_flat: pt.Tensor,
    ) -> pt.Tensor:
        c, h, w = self._infer_image_shape()
        return img_flat.reshape(img_flat.shape[0], c, h, w)

    def _image_to_flat_with_basis(
        self,
        *,
        basis: Optional[pt.Tensor],
        img: pt.Tensor,
    ) -> pt.Tensor:
        img_flat = img.reshape(img.shape[0], -1)
        if basis is None:
            return img_flat
        return pt.cat([basis, img_flat], dim=1)

    def _d4_transforms(
        self,
        *,
        x_img: pt.Tensor,
    ) -> List[pt.Tensor]:
        # D4: 4 rotations + 4 rotations of a reflected copy.
        # Use a single reflection generator: horizontal flip (W axis).
        rots = [pt.rot90(x_img, k=k, dims=(2, 3)) for k in (0, 1, 2, 3)]
        return rots

        # if not self.symmetry_use_d4:
        #     return rots

        # x_ref = pt.flip(x_img, dims=(3,))  # reflect along width
        # rots_ref = [pt.rot90(x_ref, k=k, dims=(2, 3)) for k in (0, 1, 2, 3)]
        # return rots + rots_ref

    def _all_symmetry_transforms(
        self,
        *,
        x_img: pt.Tensor,
    ) -> List[pt.Tensor]:
        imgs = self._d4_transforms(x_img=x_img)
        if self.symmetry_use_spinflip:
            imgs = imgs + [(-img) for img in imgs]
        return imgs

    def _forward_single(
        self,
        *,
        x_flat: pt.Tensor,
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        embeddings = self.backbone(x_flat, dtype=self.dtype)
        logits = self.classification_head(embeddings)

        temperature = self.get_temperature()
        logits = logits / temperature
        return logits, embeddings

    def forward(self, x: pt.Tensor):
        # If symmetry pooling is disabled, keep your original behavior.
        if not self.symmetry_pooling:
            logits, embeddings = self._forward_single(x_flat=x)
            return logits, embeddings

        basis, img_flat = self._split_basis_and_image_flat(x_flat=x)
        x_img = self._flat_to_image(img_flat=img_flat)

        x_imgs = self._all_symmetry_transforms(x_img=x_img)

        probs_accum = None
        emb_accum = None

        for img_t in x_imgs:
            x_flat_t = self._image_to_flat_with_basis(basis=basis, img=img_t)
            logits_t, emb_t = self._forward_single(x_flat=x_flat_t)

            probs_t = pt.softmax(logits_t, dim=1)

            probs_accum = probs_t if probs_accum is None else (probs_accum + probs_t)
            if self.symmetry_pool_embeddings:
                emb_accum = emb_t if emb_accum is None else (emb_accum + emb_t)

        probs_mean = probs_accum / float(len(x_imgs))
        logits_pooled = pt.log(probs_mean.clamp(min=self.symmetry_eps))

        if self.symmetry_pool_embeddings:
            embeddings_pooled = emb_accum / float(len(x_imgs))
        else:
            # Still return something reasonable
            embeddings_pooled = self.backbone(x, dtype=self.dtype)

        return logits_pooled, embeddings_pooled

    def training_step(self, batch, batch_idx):
        snapshots, labels = batch
        logits, _ = self.forward(snapshots)
        loss = self.criterion(logits, labels.argmax(dim=1))
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train_acc",
            (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean(),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        snapshots, labels = batch
        logits, _ = self.forward(snapshots)
        loss = self.criterion(logits, labels.argmax(dim=1))
        acc = (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return pt.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# import torch as pt
# import pytorch_lightning as pl

# import pandas as pd

# from abc import ABC, abstractmethod


# class ClassificationBackbone(pt.nn.Module, ABC):    
#     call_super_init = True
#     """Abstract base class for classification backbones."""
#     def __init__(self, *args, embedding_size: int = None, **kwargs):
#         """
#         Initialize the backbone.
        
#         Args:
#             embedding_size: Size of the embedding space that the backbone will output
#         """
#         super().__init__(*args, **kwargs)
#         assert embedding_size is not None, "embedding_size must be provided"
#         self.embedding_size = embedding_size

#     @abstractmethod
#     def forward(self, x: pt.Tensor, dtype: pt.dtype = None) -> pt.Tensor:
#         """
#         Forward pass of the backbone.
        
#         Args:
#             x: Input tensor of shape (batch_size, input_size)
            
#         Returns:
#             Tensor of shape (batch_size, embedding_size)
#         """
#         pass


# class MetricsLoggerCallback(pl.Callback):
#     def __init__(self):
#         super().__init__()
#         self.metrics = {}

#     def on_train_epoch_end(self, trainer, pl_module):
#         # Collect metrics logged during the training epoch
#         metrics = trainer.callback_metrics
#         for key in metrics:
#             if 'train' in key:
#                 if key not in self.metrics:
#                     self.metrics[key] = []
#                 self.metrics[key].append(metrics[key].item())

#     def on_validation_epoch_end(self, trainer, pl_module):
#         # Collect metrics logged during the training epoch
#         metrics = trainer.callback_metrics
#         for key in metrics:
#             if 'val' in key:
#                 if key not in self.metrics:
#                     self.metrics[key] = []
#                 self.metrics[key].append(metrics[key].item())
    
#     def to_pandas(self):
#         # robust to different lengths (e.g. if val isn’t run every epoch)
#         df = pd.DataFrame({k: pd.Series(v) for k, v in self.metrics.items()})
#         df.index.name = "epoch"
#         df = df.reset_index()

#         # optional: nice column order if present
#         preferred = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
#         cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
#         return df[cols]


# # LightningModule: Defines the model and training/validation steps
# class ClassificationLightningModule(pl.LightningModule):
#     def __init__(self, 
#                  backbone: ClassificationBackbone,
#                  num_classes: int,
#                  learning_rate: float = 0.001):
#         """
#         Initialize the classification module.
        
#         Args:
#             backbone: A ClassificationBackbone that takes input of shape (batch_size, input_size)
#                      and outputs embeddings of shape (batch_size, embedding_size)
#             num_classes: Number of classes for classification
#             learning_rate: Learning rate for optimization
#         """
#         super().__init__()
#         self.save_hyperparameters(ignore=["backbone"])
        
#         # Store and register the backbone model
#         if not isinstance(backbone, ClassificationBackbone):
#             raise TypeError("backbone must be a ClassificationBackbone")
#         self.backbone = backbone
        
#         # Create and register the classification head
#         self.classification_head = pt.nn.Linear(backbone.embedding_size, num_classes)
#         self.criterion = pt.nn.CrossEntropyLoss()

#         # --- Temperature parameter (frozen by default) ---
#         # We store log_temperature to enforce positivity via exp().
#         # Initialize to 0 => temperature = 1.
#         self.log_temperature = pt.nn.Parameter(pt.zeros(1), requires_grad=False)

#     def get_temperature(self) -> pt.Tensor:
#         # Temperature is strictly positive
#         # Optional: clamp to avoid crazy extremes in calibration
#         temperature = pt.exp(self.log_temperature)
#         return temperature.clamp(min=1e-3, max=1e3)

#     def enable_temperature_calibration(self):
#         self.log_temperature.requires_grad_(True)

#     def disable_temperature_calibration(self):
#         self.log_temperature.requires_grad_(False)
       
#     def forward(self, x):
#         # Get embeddings from the backbone
#         embeddings = self.backbone(x, dtype=self.dtype)

#         # Get logits from classification head
#         logits = self.classification_head(embeddings)
#         temperature = self.get_temperature()
#         logits = logits / temperature

#         return logits, embeddings

#     def training_step(self, batch, batch_idx):
#         snapshots, labels = batch
#         self.train()
#         logits, _ = self.forward(snapshots)
#         loss = self.criterion(logits, labels.argmax(dim=1))
#         self.log("train_loss", loss, on_step=False, on_epoch=True,)
#         self.log("train_acc", (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean(), on_step=False, on_epoch=True,)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         snapshots, labels = batch
#         self.eval()
#         logits, _ = self.forward(snapshots)
#         loss = self.criterion(logits, labels.argmax(dim=1))
#         acc = (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
#         self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
#         return loss

#     def configure_optimizers(self):
#         return pt.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

