import torch as pt
import pytorch_lightning as pl

from .feedforward_classification import ClassificationLightningModule

class TemperatureCalibrationLightningModule(pl.LightningModule):
    def __init__(self, base_model: ClassificationLightningModule, learning_rate_temperature: float = 1e-3):
        super().__init__()
        self.base_model = base_model
        self.learning_rate_temperature = learning_rate_temperature

        # Freeze everything except log_temperature
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Make sure temperature exists and is enabled
        assert hasattr(self.base_model, "log_temperature"), "Base model must define log_temperature"
        self.base_model.enable_temperature_calibration()
        self.criterion = pt.nn.CrossEntropyLoss()

    def forward(self, snapshots):
        # Run base model forward; backbone/head should be eval-like (no dropout/bn updates)
        self.base_model.backbone.eval()
        self.base_model.classification_head.eval()
        logits, _ = self.base_model(snapshots)
        return logits

    def training_step(self, batch, batch_idx):
        snapshots, labels = batch
        self.train()
        logits = self.forward(snapshots)
        loss = self.criterion(logits, labels.argmax(dim=1))
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        snapshots, labels = batch
        self.eval()
        logits = self.forward(snapshots)
        loss = self.criterion(logits, labels.argmax(dim=1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Only optimize the scalar temperature
        return pt.optim.Adam([self.base_model.log_temperature], lr=self.learning_rate_temperature)

    def on_fit_end(self):
        # Unfreeze everything except log_temperature
        for p in self.base_model.parameters():
            p.requires_grad_(True)
        # Lock temperature after calibration
        self.base_model.disable_temperature_calibration()
