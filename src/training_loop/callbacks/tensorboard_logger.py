from __future__ import annotations

from typing import Literal

from .callback import Callback


class TensorBoardLogger(Callback):

    def __init__(self,
                 logdir: str | None = None,
                 update_freq: int = 1,
                 update_freq_unit: Literal['epoch', 'batch'] = 'epoch'):
        super().__init__()
        self._logdir = logdir
        self._update_freq = update_freq
        self._update_freq_unit = update_freq_unit

    def on_training_begin(self):
        from torch.utils import tensorboard

        self._writer = tensorboard.SummaryWriter(self._logdir)
        self._wait = 0
        self._global_step = 0

    def on_train_batch_end(self, batch: int, logs: dict[str, float]):
        if self._update_freq_unit == 'epoch':
            return

        if self._wait == 0:
            self._writer.add_scalars(
                'batch',
                logs,
                global_step=self._global_step,
            )

    def on_val_batch_end(self, batch: int, logs: dict[str, float]):
        if self._update_freq_unit == 'epoch':
            return

        if self._wait == 0:
            self._writer.add_scalars(
                'batch',
                logs,
                global_step=self._global_step,
            )

        # Increase the counters for both train/val batch.
        self._update_counters()

    def on_epoch_end(self, epoch: int, logs: dict[str, float]):
        if self._update_freq_unit == 'batch':
            return

        if self._wait == 0:
            self._writer.add_scalars(
                'epoch',
                logs,
                global_step=self._global_step,
            )

        self._update_counters()

    def on_training_end(self):
        self._writer.close()

    def _update_counters(self):
        self._wait = (self._wait + 1) % self._update_freq
        self._global_step += 1
