from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback, TuneCallback
from typing import Optional, Union, List, Dict


class CustomReportCheckpointCallback(TuneReportCheckpointCallback):
    _allowed = TuneCallback._allowed + ["train_epoch_end"]

    def __init__(self, metrics: Optional[Union[str, List[str], Dict[str, str]]] = None, filename: str = "checkpoint",
                 on: Union[str, List[str]] = "validation_end"):
        super().__init__(metrics, filename, 'train_end')
        self._on = self._on + [on]

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "train_epoch_end" in self._on:
            self._handle(trainer, pl_module)
