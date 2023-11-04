from __future__ import annotations

from functools import wraps
from itertools import chain
import logging
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Generic, Tuple, List

from .distributed_training_step import DistributedTrainingStep
from .utils import (
    TRAIN_DATALOADER_SEPARATOR,
    train_dataloader_separator,
    prefix_val_metrics_keys,
)
from ..callbacks.callback import Callback
from ..exceptions import StopTraining
from ..types import TData

_LOGGER = logging.getLogger('DistributedTrainingLoop')
_VAL_METRICS_PREFIX = 'val_'


def _execute_on_main_process(func):
    """
    A convenient decorator for marking a method should only be executed on main process.
    """

    @wraps(func)
    def on_main_process_only(self: DistributedTrainingLoop, *args, **kwargs):
        if self._is_main_process:
            return func(self, *args, **kwargs)

    return on_main_process_only


class DistributedTrainingLoop(Generic[TData]):

    # The main process in which: callbacks are called, metrics are synced and stored,
    # and where stop training signal is broadcasted from.
    _MAIN_PROCESS = 0

    def __init__(self, model: DDP, step: DistributedTrainingStep[TData], *,
                 rank: int, device: int | None) -> None:
        self._model = model
        self._step = step
        self._rank = rank
        self._device = device

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        *,
        epochs: int,
        callbacks: List[Callback[DDP]] | None = None,
        average_metrics_after_batch_end: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_history = []
        val_history = []

        step = self._step
        model = self.model
        device = self.device

        # If we are not on the main process, callbacks are ignored
        # and no need to store them.
        if (callbacks is None) or (not self._is_main_process):
            callbacks = []

        # Training Start.
        step.init_distributed(model, device)
        self._init_callbacks(callbacks)
        self._handle(callbacks, 'training_begin')

        total_batches = len(train_dataloader) + len(val_dataloader) + 1

        for epoch in range(1, epochs + 1):
            ## Epoch Start.
            self._handle(callbacks, 'epoch_begin', epoch=epoch)
            step.reset_train_metrics_distributed()
            step.reset_val_metrics_distributed()

            dataloader = chain(
                enumerate(train_dataloader, start=1),
                train_dataloader_separator(),
                enumerate(val_dataloader, start=1),
            )

            if self._is_main_process:
                progress_bar = tqdm(dataloader, total=total_batches)
                progress_bar.set_description(
                    f'Epoch {epoch}/{epochs} - Training')

                dataloader = progress_bar
            else:
                progress_bar = None

            is_training = True
            for batch, data in dataloader:
                if data is TRAIN_DATALOADER_SEPARATOR:
                    is_training = False

                    if progress_bar is not None:
                        progress_bar.set_description(
                            f'Epoch {epoch}/{epochs} - Validating')
                    continue

                self._handle(
                    callbacks,
                    'train_batch_begin' if is_training else 'val_batch_begin',
                    batch=batch)

                if is_training:
                    logs = step.train_step_distributed(model=self._model,
                                                       data=data,
                                                       device=self._device)
                else:
                    with torch.no_grad():
                        logs = step.val_step_distributed(model=self._model,
                                                         data=data,
                                                         device=self._device)
                        logs = prefix_val_metrics_keys(logs,
                                                       _VAL_METRICS_PREFIX)

                self._handle(
                    callbacks,
                    'train_batch_end' if is_training else 'val_batch_end',
                    batch=batch,
                    logs=logs)

                if average_metrics_after_batch_end:
                    logs = self._sync_and_avg_metrics(logs)

                if progress_bar is not None:
                    # Display progress.
                    progress_bar.set_postfix(logs)

                if self._is_main_process:
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
                **step.compute_train_metrics_synced(),
                **prefix_val_metrics_keys(step.compute_val_metrics_synced(), _VAL_METRICS_PREFIX),
            }

            stop_training = self._handle(
                callbacks,
                'epoch_end',
                epoch=epoch,
                logs=logs,
            )
            stop_training = self._broadcast_stop_training(stop_training)

            # Update progress bar.
            if progress_bar is not None:
                progress_bar.set_description(
                    f'Epoch {epoch}/{epochs} - Finished')
                progress_bar.set_postfix(logs)

            # Record history.
            if self._is_main_process:
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

            # Stop training if signal was raised.
            if stop_training:
                _LOGGER.info(
                    f'Stop training at epoch {epoch} on process {self._device} '
                    'due to `StopTraining` raised.')
                break

            ## Epoch End.

        # Training End.
        self._handle(callbacks, 'training_end')

        return (pd.DataFrame(train_history).set_index(
            ['epoch', 'batch'],
            drop=False,
        ), pd.DataFrame(val_history).set_index(
            ['val_epoch', 'val_batch'],
            drop=False,
        )) if self._is_main_process else None

    @property
    def _is_main_process(self):
        return self._rank == self._MAIN_PROCESS

    @_execute_on_main_process
    def _init_callbacks(self, callbacks: list[Callback[DDP]]):
        for callback in callbacks:
            callback.set_training_loop(self)

    @_execute_on_main_process
    def _handle(
        self,
        callbacks: list[Callback[DDP]],
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

    def _broadcast_stop_training(self, stop_training: bool | None) -> bool:
        signal = torch.zeros(1, dtype=torch.int, device=self._device)

        if stop_training:
            signal += 1

        dist.broadcast(signal, src=self._MAIN_PROCESS)
        return signal.cpu().item() == 1

    def _sync_and_avg_metrics(self, metrics: Dict[str,
                                                  float]) -> Dict[str, float]:
        values = np.asarray(list(metrics.values())) / dist.get_world_size()
        values = torch.tensor(values, device=self._device)

        values = dist.reduce(values, self._MAIN_PROCESS, dist.ReduceOp.SUM)

        if values is not None:
            return {k: v for k, v in zip(metrics.keys(), values)}

        # Just return the original metrics unchanged.
        return metrics
