from __future__ import annotations

import numpy as np
from pathlib import Path
import torch
from torch import nn
from typing import Callable, Literal

from .callback import Callback


class ModelCheckpoint(Callback[nn.Module]):

    def __init__(
        self,
        filepath: str | Path | Callable[[int, dict[str, float]], str | Path],
        *,
        save_weights_only: bool = True,
        save_best_only: bool = True,
        monitor: str | None = 'val_loss',
        mode: Literal['min', 'max'] | None = 'min',
    ):
        """
        Save model at the end of every epoch.

        Parameters:
            filepath: str or Path or a function (int, dict) -> str | Path
                Path to the location to save the model.
            save_weights_only: bool
                Whether a full model will be saved (weights and its structure) or
                just the model's weights. In the case of full model save,
                `torch.save(model, path)` will be used. Otherwise,
                `torch.save(model.state_dict(), path)` will be used.
                Default to True.
            save_best_only: bool
                Whether to save the best model only, default to True.
                If this is False, the model will be saved at the end of each epoch.
            monitor: str
                Which metric to use for determining the best model,
                default to `val_loss`. This is ignored when `save_best_only = False`.
            mode: 'min' or 'max'
                How should we determine the best model, default to `min`.
                This is ignored when `save_best_only = False`.
        """
        super().__init__()
        self._filepath = filepath
        self._save_weights_only = save_weights_only

        if save_best_only:
            assert monitor is not None and mode is not None

        self._save_best_only = save_best_only
        self._monitor = monitor
        self._mode = mode

    def on_training_begin(self):
        self._best_value = np.inf if self._mode == 'min' else -np.inf

    def on_epoch_end(self, epoch: int, logs: dict[str, float]):
        if not self._save_best_only:
            self._save_model(epoch, logs)
        else:
            if self._monitor not in logs:
                raise ValueError(
                    f'Metric {self._monitor} doesnt exist in logs.')

            value = logs[self._monitor]
            if self._is_better_than_best_value(value):
                self._best_value = value
                self._save_model(epoch, logs)

    def _save_model(self, epoch: int, logs: dict[str, float]):
        if callable(self._filepath):
            filepath = self._filepath(epoch, logs)
        else:
            filepath = self._filepath

        if self._save_weights_only:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.model, filepath)

    def _is_better_than_best_value(self, value):
        if self._mode == 'min':
            return value <= self._best_value
        else:
            return value >= self._best_value
