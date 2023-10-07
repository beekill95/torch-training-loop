from __future__ import annotations

import abc
from itertools import chain
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Any, Generic

from ..callbacks import Callback
from ..exceptions import StopTraining
from ..types import TModel

_LOGGER = logging.getLogger('TrainingLoop')
_VAL_METRICS_PREFIX = 'val_'


class TrainingLoop(Generic[TModel], abc.ABC):

    def __init__(
        self,
        model: TModel,
        *,
        device: str | torch.device = 'cpu',
    ) -> None:
        """
        Base class for a training loop.
        """
        super().__init__()

        self._model = model.to(device)
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        *,
        epochs: int,
        callbacks: list[Callback[TModel]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        A skeleton for training and validating a typical model on
        the corresponding train/val datasets. This function will handle
        calling `train_step()`, `val_step()`, and calling callbacks' events.
        Moreover, it will also handle displaying training progress using
        `tqdm`. In order for it to display the progress, subclasses should
        override `train_step()` and `val_step()` and return losses and metrics
        to be displayed in the progress bar. Furthermore, subclasses should
        override the `reset_*_metrics()` and `compute_*_metrics()` in order to
        reset and compute metrics at the beginning and the end of each epoch.

        Parameters:
            train_dataloader: DataLoader
                Dataloader for loading training dataset. It should implement `__len__()`
                method in order to display the training progress.
            val_dataloader: DataLoader
                Dataloader for loading validation dataset. It should implement `__len__()`
                method in order to display the validation progress.
            epochs: int
                Number of epochs to train the model.
            callbacks: a list of callbacks or None.
                A list of callbacks to handle train/validation events, default is None.

        Returns: (pd.DataFrame, pd.DataFrame)
            A tuple of pandas dataframes containing training and validation history
            (i.e, the metrics results after each batch and epoch), respectively.
            The key of each dataframe is (epoch, batch). Rows with batch = None are
            the metrics/losses for the whole epochs.
        """
        train_history = []
        val_history = []

        if callbacks is None:
            callbacks = []

        self._init_callbacks(callbacks)
        self._handle(callbacks, 'training_begin')

        total_batches = len(train_dataloader) + len(val_dataloader) + 1

        for epoch in range(epochs):
            # Start an epoch.
            self._handle(callbacks, 'epoch_begin', epoch=epoch)
            self.reset_train_metrics()
            self.reset_val_metrics()

            dataloader = chain(
                enumerate(train_dataloader),
                _train_dataloader_separator(),
                enumerate(val_dataloader),
            )

            # Display progress bar.
            progress_bar = tqdm(dataloader, total=total_batches)
            progress_bar.set_description(f'Epoch {epoch}/{epochs} - Training')

            is_training = True
            for batch, data in progress_bar:
                # Transition to validation.
                if data is _TRAIN_DATALOADER_SEPARATOR:
                    is_training = False
                    progress_bar.set_description(
                        f'Epoch {epoch}/{epochs} - Validating')
                    continue

                self._handle(
                    callbacks,
                    'train_batch_begin' if is_training else 'val_batch_begin',
                    batch=batch)

                if is_training:
                    logs = self.train_step(data)
                else:
                    with torch.no_grad():
                        logs = _prefix_val_metrics_keys(self.val_step(data))

                self._handle(
                    callbacks,
                    'train_batch_end' if is_training else 'val_batch_end',
                    batch=batch,
                    logs=logs)

                # Display progress.
                progress_bar.set_postfix(logs)

                # Record progress history.
                history = train_history if is_training else val_history
                history.append({
                    **logs,
                    'batch': batch,
                    'epoch': epoch,
                })

            # End epoch.
            logs = {
                **self.compute_train_metrics(),
                **_prefix_val_metrics_keys(self.compute_val_metrics()),
            }

            stop_training = self._handle(
                callbacks,
                'epoch_end',
                epoch=epoch,
                logs=logs,
            )

            # Record history.
            train_history.append({
                **{
                    k: v
                    for k, v in logs.items() if ~k.startswith(_VAL_METRICS_PREFIX)
                },
                'epoch': epoch,
                'batch': None,
            })
            val_history.append({
                **{
                    k: v
                    for k, v in logs.items() if k.startswith(_VAL_METRICS_PREFIX)
                },
                'epoch': epoch,
                'batch': None,
            })

            # Stop training if a signal was raised.
            if stop_training:
                _LOGGER.info(
                    f'Stop training at epoch {epoch} due `StopTraining` raised.'
                )
                break

        self._handle(callbacks, 'training_end')
        return tuple(
            pd.DataFrame(history).set_index(['epoch', 'batch'], drop=False)
            for history in [train_history, val_history])

    @abc.abstractmethod
    def train_step(self, data: Any) -> dict[str, float]:
        """
        Perform one train step over the given data. Subclasses
        should implement this method to perform feed-forward
        and back-propagation. Moreover, the function should return
        a dictionary of metrics and their values to display the
        training progress.

        Parameters:
            data: Any
                A mini-batch returned by the train dataloader.

        Returns: dict[str, float]
            Train metrics to be displayed in the progress bar.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def val_step(self, data: Any) -> dict[str, float]:
        """
        Perform one validation over the given data. Subclasses
        should implement this method to perform feed-forward
        over the data. The function should return a dictionary
        of metrics and their values to display the validation progress.
        The returned dictionary's keys will be prefixed with 'val_'
        before displaying, unless they already have that prefix.

        Parameters:
            data: Any
                A mini-batch returned by the validation dataloader.

        Returns: dict[str, float]
            Validation metrics to be displayed in the progress bar.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def reset_train_metrics(self):
        """Reset training metrics, will be used at the start of an epoch."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def reset_val_metrics(self):
        """Reset validation metrics, will be used at the start of an epoch."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def compute_train_metrics(self) -> dict[str, float]:
        """
        Compute training metrics, will be used at the end of an epoch.

        Returns: dict[str, float]
            Training metrics to be displayed in the progress bar at the end of an epoch.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def compute_val_metrics(self) -> dict[str, float]:
        """
        Compute validation metrics, will be used at the end of an epoch.

        Returns: dict[str, float]
            Validation metrics to be displayed in the progress bar
            at the end of an epoch.
        """
        pass

    def _init_callbacks(self, callbacks: list[Callback[TModel]]):
        for callback in callbacks:
            callback.set_model(self.model)

    def _handle(
        self,
        callbacks: list[Callback[TModel]],
        event: str,
        **kwargs,
    ) -> bool:
        stop_training = False

        for callback in callbacks:
            try:
                callback.on(event, **kwargs)
            except StopTraining:
                stop_training = True

        return stop_training


_TRAIN_DATALOADER_SEPARATOR = ()


def _train_dataloader_separator():
    yield (-1, _TRAIN_DATALOADER_SEPARATOR)


def _prefix_val_metrics_keys(metrics: dict[str, float]) -> dict[str, float]:

    def prefix_key(key):
        return (key if key.startswith(_VAL_METRICS_PREFIX) else
                f'{_VAL_METRICS_PREFIX}{key}')

    return {prefix_key(k): v for k, v in metrics.items()}
