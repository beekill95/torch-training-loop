from __future__ import annotations

from itertools import chain
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Generic

from .training_step import TrainingStep
from ..callbacks import Callback
from ..exceptions import StopTraining
from ..types import TData, TModel

_LOGGER = logging.getLogger('TrainingLoop')
_VAL_METRICS_PREFIX = 'val_'


class TrainingLoop(Generic[TModel, TData]):

    def __init__(
        self,
        model: TModel,
        step: TrainingStep[TModel, TData],
        *,
        device: str | torch.device = 'cpu',
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

        total_batches = len(train_dataloader) + len(val_dataloader) + 1

        for epoch in range(1, epochs + 1):
            ## Epoch Start.
            self._handle(callbacks, 'epoch_begin', epoch=epoch)
            step.reset_train_metrics()
            step.reset_val_metrics()

            dataloader = chain(
                enumerate(train_dataloader, start=1),
                _train_dataloader_separator(),
                enumerate(val_dataloader, start=1),
            )

            # Display progress bar.
            progress_bar = tqdm(dataloader, total=total_batches)
            progress_bar.set_description(f'Epoch {epoch}/{epochs} - Training')

            is_training = True
            for batch, data in progress_bar:
                ## Batch Start.

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
                    logs = step.train_step(model, data, device)
                else:
                    with torch.no_grad():
                        logs = step.val_step(model, data, device)
                        logs = _prefix_val_metrics_keys(logs)

                self._handle(
                    callbacks,
                    'train_batch_end' if is_training else 'val_batch_end',
                    batch=batch,
                    logs=logs)

                # Display progress.
                progress_bar.set_postfix(logs)

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

                ## Batch End.

            # Gather training and validation logs when an epoch ends.
            logs = {
                **step.compute_train_metrics(),
                **_prefix_val_metrics_keys(step.compute_val_metrics()),
            }

            stop_training = self._handle(
                callbacks,
                'epoch_end',
                epoch=epoch,
                logs=logs,
            )

            # Update progress bar.
            progress_bar.set_description(f'Epoch {epoch}/{epochs} - Finished')
            progress_bar.set_postfix(logs)

            # Record history.
            train_history.append({
                **{
                    k: v
                    for k, v in logs.items() if not k.startswith(_VAL_METRICS_PREFIX)
                },
                'epoch': epoch,
                'batch': -1,
            })
            val_history.append({
                **{
                    k: v
                    for k, v in logs.items() if k.startswith(_VAL_METRICS_PREFIX)
                },
                'val_epoch': epoch,
                'val_batch': -1,
            })

            # Stop training if a signal was raised.
            if stop_training:
                _LOGGER.info(
                    f'Stop training at epoch {epoch} due `StopTraining` raised.'
                )
                break

            ## Epoch End.

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


_TRAIN_DATALOADER_SEPARATOR = ()


def _train_dataloader_separator():
    yield (-1, _TRAIN_DATALOADER_SEPARATOR)


def _prefix_val_metrics_keys(metrics: dict[str, float]) -> dict[str, float]:

    def prefix_key(key):
        return (key if key.startswith(_VAL_METRICS_PREFIX) else
                f'{_VAL_METRICS_PREFIX}{key}')

    return {prefix_key(k): v for k, v in metrics.items()}
