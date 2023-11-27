from __future__ import annotations

import logging
from itertools import chain
from typing import Generic

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..callbacks import Callback
from ..exceptions import StopTraining
from ..progress_reporter import ProgressReporter
from ..types import TData
from ..types import TDevice
from ..types import TModel
from .training_step import TrainingStep
from .utils import prefix_val_metrics_keys
from .utils import TRAIN_DATALOADER_SEPARATOR
from .utils import train_dataloader_separator

_LOGGER = logging.getLogger('TrainingLoop')
_VAL_METRICS_PREFIX = 'val_'


class TrainingLoop(Generic[TModel, TData]):

    def __init__(
        self,
        model: TModel,
        step: TrainingStep[TModel, TData],
        *,
        device: TDevice = 'cpu',
    ) -> None:
        """
        Base class for a training loop.
        """
        super().__init__()

        self._model = model.to(device)
        self._device = device
        self._step = step

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
        verbose: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        A skeleton for training and validating a typical model on
        the corresponding train/val datasets. This function will handle
        calling `train_step()`, `val_step()` of the trainig step class,
        and calling callbacks' events. Moreover, it will also handle displaying
        training progress using `tqdm`. In order for it to display the progress,
        training step should override `train_step()` and `val_step()` and return losses
        and metrics to be displayed in the progress bar. Furthermore, training step
        subclasses should override the `reset_*_metrics()` and `compute_*_metrics()` in
        order to reset and compute metrics at the beginning and the end of each epoch.

        Parameters:
            train_dataloader: DataLoader
                Dataloader for loading training dataset.
            val_dataloader: DataLoader
                Dataloader for loading validation dataset.
            epochs: int
                Number of epochs to train the model.
            callbacks: a list of callbacks or None.
                A list of callbacks to handle train/validation events, default is None.
            verbose: int
                Verbose level. Default to 1.
                If verbose < 1: no progress bar is displayed.
                If verbose = 1: progress bar is displayed at each epoch.
                If verbose = 2: no progress bar is displayed, but train and validation
                    metrics are displayed after each epoch.
                If verbose > 2: number of epochs between consecutive reports
                    of train and validation metrics.

        Returns: (pd.DataFrame, pd.DataFrame)
            A tuple of pandas dataframes containing training and validation history
            (i.e, the metrics results after each batch and epoch), respectively.
            The key of each dataframe is (epoch, batch). Rows with batch = -1 are
            the metrics/losses for the whole epochs.
        """
        train_history = []
        val_history = []

        step = self._step
        model = self.model
        device = self.device

        if callbacks is None:
            callbacks = []

        step.init(model, device)
        self._init_callbacks(callbacks)
        self._handle(callbacks, 'training_begin')

        try:
            total_batches = len(train_dataloader) + len(val_dataloader)
        except TypeError:
            total_batches = float('inf')

        for epoch in range(1, epochs + 1):
            # Epoch Start.
            self._handle(callbacks, 'epoch_begin', epoch=epoch)
            step.reset_train_metrics()
            step.reset_val_metrics()

            dataloader = chain(
                enumerate(train_dataloader, start=1),
                train_dataloader_separator(),
                enumerate(val_dataloader, start=1),
            )

            # Display progress bar.
            with ProgressReporter(
                    epoch,
                    total_epochs=epochs,
                    total_batches=total_batches,
                    verbose=verbose,
            ) as reporter:
                is_training = True
                for batch, data in dataloader:
                    # Transition to validation.
                    if data is TRAIN_DATALOADER_SEPARATOR:
                        is_training = False
                        continue

                    # Batch Start.
                    reporter.next_batch()

                    self._handle(
                        callbacks,
                        'train_batch_begin' if is_training else 'val_batch_begin',
                        batch=batch)

                    if is_training:
                        logs = step.train_step(model, data, device)
                    else:
                        with torch.no_grad():
                            logs = step.val_step(model, data, device)
                            logs = prefix_val_metrics_keys(logs, _VAL_METRICS_PREFIX)

                    self._handle(
                        callbacks,
                        'train_batch_end' if is_training else 'val_batch_end',
                        batch=batch,
                        logs=logs)

                    # Display progress.
                    reporter.report_batch_progress(
                        'Training' if is_training else 'Validating', logs)

                    # Record progress history.
                    if is_training:
                        train_history.append({
                            **logs,
                            'batch': batch,
                            'epoch': epoch,
                        })
                    else:
                        val_history.append({
                            **logs,
                            'val_batch': batch,
                            'val_epoch': epoch,
                        })

                    # Batch End.

                # Gather training and validation logs when an epoch ends.
                logs = {
                    **step.compute_train_metrics(),
                    **prefix_val_metrics_keys(step.compute_val_metrics(),
                                              _VAL_METRICS_PREFIX),
                }

                stop_training = self._handle(
                    callbacks,
                    'epoch_end',
                    epoch=epoch,
                    logs=logs,
                )

                # Update progress bar.
                reporter.report_epoch_progress('Finished', logs)

                # Record history.
                train_history.append({
                    **{
                        k: v
                        for k, v in logs.items()
                        if not k.startswith(_VAL_METRICS_PREFIX)
                    },
                    'epoch': epoch,
                    'batch': -1,
                })
                val_history.append({
                    **{
                        k: v
                        for k, v in logs.items()
                        if k.startswith(_VAL_METRICS_PREFIX)
                    },
                    'val_epoch': epoch,
                    'val_batch': -1,
                })

                # Stop training if a signal was raised.
                if stop_training:
                    _LOGGER.info(
                        f'Stop training at epoch {epoch} due `StopTraining` raised.')
                    break

                # Epoch End.

        self._handle(callbacks, 'training_end')

        return (pd.DataFrame(train_history).set_index(
            ['epoch', 'batch'],
            drop=False,
        ), pd.DataFrame(val_history).set_index(
            ['val_epoch', 'val_batch'],
            drop=False,
        ))

    def _init_callbacks(self, callbacks: list[Callback[TModel]]):
        for callback in callbacks:
            callback.set_training_loop(self)

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
