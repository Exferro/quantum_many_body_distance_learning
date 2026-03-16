import pytorch_lightning as pl


class FDivergenceCallback(pl.Callback):
    def __init__(
            self,
            pipeline=None,
            training_or_calibration: str = 'training',
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        assert training_or_calibration in ('training', 'calibration')
        self.training_or_calibration = training_or_calibration

    def on_train_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
    ) -> None:
        current_epoch = trainer.current_epoch
        if self.training_or_calibration == 'training':
            assert len(self.pipeline.training_f_matrices) < current_epoch + 1
            self.pipeline.training_f_matrices.append(
                self.pipeline.calculate_f_divergences_matrix(divergence=self.pipeline.transient_f_divergence,
                                                             selector=self.pipeline.transient_selector)[0])
        elif self.training_or_calibration == 'calibration':
            assert len(self.pipeline.calibration_f_matrices) < current_epoch + 1
            self.pipeline.calibration_f_matrices.append(
                self.pipeline.calculate_f_divergences_matrix(divergence=self.pipeline.transient_f_divergence,
                                                                selector=self.pipeline.transient_selector)[0])
        else:
            raise ValueError(f'Unknown training_or_calibration value: {self.training_or_calibration}') 
