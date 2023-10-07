from __future__ import annotations

from copy import deepcopy
import logging
import numpy as np
from torch import nn
from typing import Literal

from .callback import Callback
from ..exceptions import StopTraining

_LOGGER = logging.getLogger('EarlyStopping')


class EarlyStopping(Callback[nn.Module]):

    def __init__(
        self,
        monitor: str,
        mode: Literal['min', 'max'],
        patience: int,
        restore_best_weights: bool = True,
    ):
        """
        Early stopping callback to prevent models overfitting.

        Parameters:
            monitor: str
                A metric in the logs given at the end of an epoch to monitor
                for early stopping.
            mode: Literal['min', 'max']
                How should the callback know if the monitored value is getting better.
            patience: int
                How many epochs should we wait before stopping the training.
            restore_best_weights: bool
                Whether should we restore the model to the best weights once the
                training ends. **Note:** unlike Keras, this flag will restore the best
                weights when the training ends due to both early stopping and the last
                epoch reached. Default to True.
        """
        super().__init__()

        self._monitor = monitor
        self._mode = mode
        self._patience = patience
        self._restore_best_weights = restore_best_weights

    def on_training_begin(self):
        self._wait = 0
        self._best_value = -np.inf if self._mode == 'max' else np.inf
        self._best_weights = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float]):
        if self._monitor not in logs:
            _LOGGER.warning(f'Metric `{self._monitor}` doesnt exist, '
                            'early stopping wont work!')
            return

        value = logs[self._monitor]
        if self._is_better_than_best_value(value):
            self._wait = 0
            self._best_value = value

            # No need to save the best weights if we're not gonna use it!
            if self._restore_best_weights:
                assert self.model is not None
                self._best_weights = deepcopy(self.model.state_dict())
        else:
            self._wait += 1

        if self._wait > self._patience:
            _LOGGER.info(
                f'Training doesnt improve model performance on {self._monitor}'
                'for the last {self._wait} epochs. Early stopping!')
            raise StopTraining()

    def on_training_end(self):
        if self._restore_best_weights and self._best_weights is not None:
            assert self.model is not None
            self.model.load_state_dict(self._best_weights)

    def _is_better_than_best_value(self, value):
        return (value >= self._best_value
                if self._mode == 'max' else value <= self._best_value)
