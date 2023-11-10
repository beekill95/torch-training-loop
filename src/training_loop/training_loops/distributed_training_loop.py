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
    hasfunc,
)
from ..callbacks.callback import Callback
from ..exceptions import StopTraining
from ..types import TData, TDevice

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
    """
    Distributed training loop that supports training distributed data parallel on
    single-node multigpus machine.
    """

    # The main process in which: callbacks are called, metrics are synced and stored,
    # and where stop training signal is broadcasted from.
    _MAIN_PROCESS = 0

    def __init__(self, model: DDP, step: DistributedTrainingStep[TData], *,
                 rank: int, device: TDevice) -> None:
        self._model = model
        self._step = step
        self._rank = rank
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
        callbacks: List[Callback[DDP]] | None = None,
        average_metrics_after_batch_end: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        A skeleton for training and validating a distributed data parallel model on
        the corresponding train/val datasets. This function will handle
        calling `train_step_distributed()`, `val_step_distributed()` of the distributed
        training step class, and calling callbacks' events. Moreover, it will also
        handle displaying training progress on the main process using `tqdm`. In order
        for it to display the progress, training step should override
        `train_step_distributed()` and `val_step_distributed()` and return losses
        and metrics to be displayed in the progress bar.
        If `average_metrics_after_batch_end` flag is True, then the class will
        average the metrics across process, otherwise, the metrics displayed will
        be from the main prrocess. Furthermore, training step subclasses should override
        the `reset_*_metrics_distributed()` and `compute_*_metrics_synced()` in
        order to reset and compute metrics at the beginning and the end of each epoch.

        Parameters:
            train_dataloader: DataLoader
                Dataloader for loading training dataset. It should implement `__len__()`
                method in order to display the training progress.
            val_dataloader: DataLoader
                Dataloader for loading validation dataset. It should implement
                `__len__()` method in order to display the validation progress.
            epochs: int
                Number of epochs to train the model.
            callbacks: a list of callbacks or None.
                A list of callbacks to handle train/validation events, default is None.
                Note: only the main process handles calling the callbacks.
            average_metrics_after_batch_end: bool
                Whether to average metrics across processes before displaying in the
                progress bar. Set the flag to True will report more accurate metrics,
                but it might come with performance hit. Default to True.

        Returns: (pd.DataFrame, pd.DataFrame) | None
            A tuple of pandas dataframes containing training and validation history
            (i.e, the metrics results after each batch and epoch), respectively.
            The key of each dataframe is (epoch, batch). Rows with batch = -1 are
            the metrics/losses for the whole epochs.
            Only on the main process does the tuple be returned. On other processes,
            None is returned.
        """
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
            # Ensure that `set_epoch` function of dataloaders are called
            # to make shuffling data of each process is correct.
            if hasfunc(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)

            if hasfunc(val_dataloader.sampler, 'set_epoch'):
                val_dataloader.sampler.set_epoch(epoch)

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

        dist.reduce(values, self._MAIN_PROCESS, dist.ReduceOp.SUM)

        if self._is_main_process:
            return {k: v for k, v in zip(metrics.keys(), values.cpu().numpy())}
        else:
            # Just return the original metrics unchanged.
            return metrics
