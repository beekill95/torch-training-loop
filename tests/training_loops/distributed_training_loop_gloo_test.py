from __future__ import annotations

from contextlib import contextmanager
import os
import pandas as pd
import pytest
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from training_loop.callbacks import Callback
from training_loop.distributed import DistributedTrainingLoop, DistributedTrainingStep
from unittest.mock import MagicMock, call, DEFAULT

from training_loop.exceptions import StopTraining

from .utils import assert_dataframes_equal


@pytest.fixture
def world_size():
    return 2


@pytest.fixture
def backend():
    return 'gloo'


@pytest.fixture
def master_port():
    with socket.socket() as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def fake_callback():
    callback = Callback()

    callback.set_training_loop = MagicMock()
    callback.on_training_begin = MagicMock()
    callback.on_training_end = MagicMock()
    callback.on_epoch_begin = MagicMock()
    callback.on_epoch_end = MagicMock()
    callback.on_train_batch_begin = MagicMock()
    callback.on_train_batch_end = MagicMock()
    callback.on_val_batch_begin = MagicMock()
    callback.on_val_batch_end = MagicMock()

    return callback


@contextmanager
def setup_backend(backend, world_size, port, rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        yield
    finally:
        os.environ.pop('MASTER_ADDR')
        os.environ.pop('MASTER_PORT')
        dist.destroy_process_group()


def _test_broadcast_stop_training_signal(rank, port, world_size, backend):
    with setup_backend(backend, world_size, port, rank):
        loop = DistributedTrainingLoop(
            MagicMock(DDP),
            step=MagicMock(DistributedTrainingStep),
            rank=rank,
            device='cpu',
        )

        stop_training = loop._broadcast_stop_training(True)
        assert stop_training


def test_broadcast_stop_training_signal(backend, world_size, master_port):
    mp.spawn(
        _test_broadcast_stop_training_signal,
        args=(master_port, world_size, backend),
        nprocs=world_size,
    )


def _test_broadcast_no_stop_training_signal(rank, port, world_size, backend):
    with setup_backend(backend, world_size, port, rank):
        loop = DistributedTrainingLoop(
            MagicMock(DDP),
            step=MagicMock(DistributedTrainingStep),
            rank=rank,
            device='cpu',
        )

        stop_training = loop._broadcast_stop_training(False)
        assert not stop_training


def test_broadcast_no_stop_training_signal(backend, world_size, master_port):
    mp.spawn(
        _test_broadcast_no_stop_training_signal,
        args=(master_port, world_size, backend),
        nprocs=world_size,
    )


def _test_sync_avg_metrics(rank, port, world_size, backend):
    metrics = {
        'loss': rank * 1.0,
        'f1': 1.0 - 0.1 * rank,
    }

    with setup_backend(backend, world_size, port, rank):
        loop = DistributedTrainingLoop(
            MagicMock(DDP),
            step=MagicMock(DistributedTrainingStep),
            rank=rank,
            device='cpu',
        )

        results = loop._sync_and_avg_metrics(metrics)

        if rank != loop._MAIN_PROCESS:
            assert results == pytest.approx(metrics)
        else:
            avg_loss = sum(i * 1.0 for i in range(world_size)) / world_size
            avg_f1 = sum(1.0 - 0.1 * i for i in range(world_size)) / world_size

            assert results['f1'] == pytest.approx(avg_f1)
            assert results['loss'] == pytest.approx(avg_loss)


def test_sync_avg_metrics(backend, world_size, master_port):
    mp.spawn(
        _test_sync_avg_metrics,
        args=(master_port, world_size, backend),
        nprocs=world_size,
    )


class TestDistributedTrainingLoop:

    train_data = ('train_1', 'train_2', 'train_3')
    val_data = ('val_1', 'val_2', 'val_3')
    train_step_return_values = (
        {
            'f1': 0.5,
            'batch': 1
        },
        {
            'f1': 0.6,
            'batch': 2
        },
        {
            'f1': 0.7,
            'batch': 3
        },
    )
    val_step_return_values = (
        {
            'f1': 0.3,
            'batch': 1
        },
        {
            'f1': 0.4,
            'batch': 2
        },
        {
            'f1': 0.5,
            'batch': 3
        },
    )

    def create_train_dataloader(self):
        dataloader = MagicMock(DataLoader)
        dataloader.sampler = MagicMock()
        dataloader.__iter__.return_value = self.train_data

        return dataloader

    def create_val_dataloader(self):
        dataloader = MagicMock(DataLoader)
        dataloader.sampler = MagicMock()
        dataloader.__iter__.return_value = self.val_data

        return dataloader

    def _test_fit_method_made_calls_with_correct_arguments(
            self, rank, port, world_size, backend):
        with setup_backend(backend, world_size, port, rank):
            step = MagicMock(DistributedTrainingStep)
            step.train_step_distributed.side_effect = self.train_step_return_values
            step.val_step_distributed.side_effect = self.val_step_return_values

            step.compute_train_metrics_synced.return_value = {
                'f1': 0.8,
                'epoch': 1
            }
            step.compute_val_metrics_synced.return_value = {
                'f1': 0.6,
                'epoch': 1
            }

            model = MagicMock(DDP)
            loop = DistributedTrainingLoop(
                model,
                step=step,
                rank=rank,
                device='cpu',
            )
            loop._sync_and_avg_metrics = MagicMock(side_effect=lambda x: x)
            loop._broadcast_stop_training = MagicMock()

            callback = fake_callback()

            trainloader = self.create_train_dataloader()
            valloader = self.create_val_dataloader()

            # Call fit.
            loop.fit(
                trainloader,
                valloader,
                epochs=1,
                callbacks=[callback],
                average_metrics_after_batch_end=True,
            )

            # Every process will call step's init method.
            step.init_distributed.assert_called_once_with(model, 'cpu')

            # Metrics stuffs will also be called on each process.
            step.reset_train_metrics_distributed.assert_called_once()
            step.reset_val_metrics_distributed.assert_called_once()
            step.compute_train_metrics_synced.assert_called_once()
            step.compute_val_metrics_synced.assert_called_once()

            # Broadcast stop signal.
            if rank == loop._MAIN_PROCESS:
                loop._broadcast_stop_training.assert_called_once_with(False)
            else:
                loop._broadcast_stop_training.assert_called_once_with(None)

            # Sync and average metrics.
            loop._sync_and_avg_metrics.assert_has_calls([
                call({
                    'f1': 0.5,
                    'batch': 1
                }),
                call({
                    'f1': 0.6,
                    'batch': 2
                }),
                call({
                    'f1': 0.7,
                    'batch': 3
                }),
                call({
                    'val_f1': 0.3,
                    'val_batch': 1
                }),
                call({
                    'val_f1': 0.4,
                    'val_batch': 2
                }),
                call({
                    'val_f1': 0.5,
                    'val_batch': 3
                }),
            ])

            # Only on the main process, the callbacks will be init and called
            # at each event.
            if rank == loop._MAIN_PROCESS:
                # Callbacks init.
                callback.set_training_loop.assert_called_once_with(loop)

                # Assert callbacks' events.
                callback.on_training_begin.assert_called_once()
                callback.on_epoch_begin.assert_called_once_with(epoch=1)

                assert callback.on_train_batch_begin.call_args_list == [
                    call(batch=1),
                    call(batch=2),
                    call(batch=3),
                ]
                assert callback.on_train_batch_end.call_args_list == [
                    call(batch=1, logs={
                        'f1': 0.5,
                        'batch': 1
                    }),
                    call(batch=2, logs={
                        'f1': 0.6,
                        'batch': 2
                    }),
                    call(batch=3, logs={
                        'f1': 0.7,
                        'batch': 3
                    }),
                ]

                assert callback.on_val_batch_begin.call_args_list == [
                    call(batch=1),
                    call(batch=2),
                    call(batch=3),
                ]
                assert callback.on_val_batch_end.call_args_list == [
                    call(batch=1, logs={
                        'val_f1': 0.3,
                        'val_batch': 1
                    }),
                    call(batch=2, logs={
                        'val_f1': 0.4,
                        'val_batch': 2
                    }),
                    call(batch=3, logs={
                        'val_f1': 0.5,
                        'val_batch': 3
                    }),
                ]

                callback.on_epoch_end.assert_called_once_with(
                    epoch=1,
                    logs={
                        'f1': 0.8,
                        'epoch': 1,
                        'val_f1': 0.6,
                        'val_epoch': 1
                    },
                )
                callback.on_training_end.assert_called_once()
            else:
                # Callbacks init.
                callback.set_training_loop.assert_not_called()

                # Assert callbacks' events.
                callback.on_training_begin.assert_not_called()
                callback.on_epoch_begin.assert_not_called()

                callback.on_train_batch_begin.assert_not_called()
                callback.on_train_batch_end.assert_not_called()

                callback.on_val_batch_begin.assert_not_called()
                callback.on_val_batch_end.assert_not_called()

                callback.on_epoch_end.assert_not_called()
                callback.on_training_end.assert_not_called()

    def test_fit_method_made_calls_with_correct_arguments(
            self, backend, world_size, master_port):
        mp.spawn(
            self._test_fit_method_made_calls_with_correct_arguments,
            args=(master_port, world_size, backend),
            nprocs=world_size,
        )

    def _test_fit_method_return_correct_histories(self, rank, port, world_size,
                                                  backend):
        with setup_backend(backend, world_size, port, rank):
            step = MagicMock(DistributedTrainingStep)
            step.train_step_distributed.side_effect = self.train_step_return_values
            step.val_step_distributed.side_effect = self.val_step_return_values

            step.compute_train_metrics_synced.return_value = {
                'f1': 0.8,
                'epoch': 1
            }
            step.compute_val_metrics_synced.return_value = {
                'f1': 0.6,
                'epoch': 1
            }

            model = MagicMock(DDP)
            loop = DistributedTrainingLoop(
                model,
                step=step,
                rank=rank,
                device='cpu',
            )
            loop._sync_and_avg_metrics = MagicMock(side_effect=lambda x: x)
            loop._broadcast_stop_training = MagicMock()

            trainloader = self.create_train_dataloader()
            valloader = self.create_val_dataloader()

            histories = loop.fit(
                trainloader,
                valloader,
                epochs=1,
            )

            if rank == loop._MAIN_PROCESS:
                train_history, val_history = histories
                assert_dataframes_equal(
                    train_history,
                    pd.DataFrame([
                        {
                            'epoch': 1,
                            'batch': 1,
                            'f1': 0.5
                        },
                        {
                            'epoch': 1,
                            'batch': 2,
                            'f1': 0.6,
                        },
                        {
                            'epoch': 1,
                            'batch': 3,
                            'f1': 0.7,
                        },
                        {
                            'epoch': 1,
                            'batch': -1,
                            'f1': 0.8,
                        },
                    ]).set_index(['epoch', 'batch'], drop=False))

                assert_dataframes_equal(
                    val_history,
                    pd.DataFrame([
                        {
                            'val_epoch': 1,
                            'val_batch': 1,
                            'val_f1': 0.3
                        },
                        {
                            'val_epoch': 1,
                            'val_batch': 2,
                            'val_f1': 0.4,
                        },
                        {
                            'val_epoch': 1,
                            'val_batch': 3,
                            'val_f1': 0.5,
                        },
                        {
                            'val_epoch': 1,
                            'val_batch': -1,
                            'val_f1': 0.6,
                        },
                    ]).set_index(['val_epoch', 'val_batch'], drop=False))
            else:
                assert histories is None

    def test_fit_method_return_correct_histories(self, world_size, backend,
                                                 master_port):
        mp.spawn(
            self._test_fit_method_return_correct_histories,
            args=(master_port, world_size, backend),
            nprocs=world_size,
        )


class TestCallsOrder:

    def record_call(self, method: str, call_orders: list):

        def record(*args, **kwargs):
            call_orders.append((method, (args, kwargs)))
            return DEFAULT

        return record

    def create_recorded_callback(self, calls: list[str]):
        callback = fake_callback()

        callback.set_training_loop.side_effect = self.record_call(
            'set_training_loop', calls)
        callback.on_training_begin.side_effect = self.record_call(
            'on_training_begin', calls)
        callback.on_training_end.side_effect = self.record_call(
            'on_training_end', calls)

        callback.on_epoch_begin.side_effect = self.record_call(
            'on_epoch_begin', calls)
        callback.on_epoch_end.side_effect = self.record_call(
            'on_epoch_end', calls)

        callback.on_train_batch_begin.side_effect = self.record_call(
            'on_train_batch_begin', calls)
        callback.on_train_batch_end.side_effect = self.record_call(
            'on_train_batch_end', calls)
        callback.on_val_batch_begin.side_effect = self.record_call(
            'on_val_batch_begin', calls)
        callback.on_val_batch_end.side_effect = self.record_call(
            'on_val_batch_end', calls)

        return callback

    def create_recorded_step(self, calls: list):
        step = MagicMock(DistributedTrainingStep)

        step.init_distributed.side_effect = self.record_call(
            'init_distributed', calls)
        step.train_step_distributed.side_effect = self.record_call(
            'train_step_distributed', calls)
        step.train_step_distributed.return_value = {'f1': 0.8}
        step.val_step_distributed.side_effect = self.record_call(
            'val_step_distributed', calls)
        step.val_step_distributed.return_value = {'f1': 0.7}

        step.compute_train_metrics_synced.side_effect = self.record_call(
            'compute_train_metrics_synced', calls)
        step.compute_train_metrics_synced.return_value = {'f1': 0.8}
        step.compute_val_metrics_synced.side_effect = self.record_call(
            'compute_val_metrics_synced', calls)
        step.compute_val_metrics_synced.return_value = {'f1': 0.7}

        step.reset_train_metrics_distributed.side_effect = self.record_call(
            'reset_train_metrics_distributed', calls)
        step.reset_val_metrics_distributed.side_effect = self.record_call(
            'reset_val_metrics_distributed', calls)

        return step

    def create_dataloader(self, return_values):
        dataloader = MagicMock(DataLoader)
        dataloader.__iter__.side_effect = lambda: iter(return_values)
        dataloader.configure_mock(sampler=None)

        return dataloader

    def _test_one_epoch(self, rank, port, world_size, backend):
        with setup_backend(backend, world_size, port, rank):
            calls = []

            callback = self.create_recorded_callback(calls)
            step = self.create_recorded_step(calls)

            loop = DistributedTrainingLoop(
                MagicMock(DDP),
                step=step,
                rank=rank,
                device='cpu',
            )

            train_loader = self.create_dataloader(
                ('train1', 'train2', 'train3', 'train4'))
            val_loader = self.create_dataloader(('val1', 'val2', 'val3'))

            loop.fit(train_loader, val_loader, epochs=1, callbacks=[callback])

            method_names = [name for name, _ in calls]
            if rank == loop._MAIN_PROCESS:
                assert method_names == [
                    'init_distributed',
                    'set_training_loop',
                    'on_training_begin',
                    # First epoch
                    'on_epoch_begin',
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    # 3 validation batches.
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    'on_epoch_end',
                    # End training.
                    'on_training_end',
                ]
            else:
                assert method_names == [
                    'init_distributed',
                    # First epoch
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    # 3 validation batches.
                    'val_step_distributed',
                    'val_step_distributed',
                    'val_step_distributed',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    # End training.
                ]

    def test_one_epoch(self, world_size, backend, master_port):
        mp.spawn(self._test_one_epoch,
                 args=(master_port, world_size, backend),
                 nprocs=world_size)

    def _test_three_epochs(self, rank, port, world_size, backend):
        with setup_backend(backend, world_size, port, rank):
            calls = []

            callback = self.create_recorded_callback(calls)
            step = self.create_recorded_step(calls)

            loop = DistributedTrainingLoop(
                MagicMock(DDP),
                step=step,
                rank=rank,
                device='cpu',
            )

            train_loader = self.create_dataloader(
                ('train1', 'train2', 'train3', 'train4'))
            val_loader = self.create_dataloader(('val1', 'val2', 'val3'))

            loop.fit(train_loader, val_loader, epochs=3, callbacks=[callback])

            method_names = [name for name, _ in calls]
            if rank == loop._MAIN_PROCESS:
                assert method_names == [
                    'init_distributed',
                    'set_training_loop',
                    'on_training_begin',
                    # First epoch
                    'on_epoch_begin',
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    # 3 validation batches.
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    'on_epoch_end',
                    # Second epoch
                    'on_epoch_begin',
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    # 3 validation batches.
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    'on_epoch_end',
                    # Third epoch
                    'on_epoch_begin',
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    # 3 validation batches.
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    'on_epoch_end',
                    # End training.
                    'on_training_end',
                ]
            else:
                assert method_names == [
                    'init_distributed',
                    # First epoch
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    # 3 validation batches.
                    'val_step_distributed',
                    'val_step_distributed',
                    'val_step_distributed',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    # Second epoch
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    # 3 validation batches.
                    'val_step_distributed',
                    'val_step_distributed',
                    'val_step_distributed',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    # Third epoch
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    # 3 validation batches.
                    'val_step_distributed',
                    'val_step_distributed',
                    'val_step_distributed',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    # End training.
                ]

    def test_three_epochs(self, world_size, backend, master_port):
        mp.spawn(self._test_three_epochs,
                 args=(master_port, world_size, backend),
                 nprocs=world_size)

    def _test_three_epochs_with_early_stopping(self, rank, port, world_size,
                                               backend):

        class EarlyStoppingAtSecondEpochEnd(Callback):

            def on_epoch_end(self, epoch: int, logs: dict[str, float]):
                if epoch == 2:
                    raise StopTraining()

        with setup_backend(backend, world_size, port, rank):
            calls = []

            callback = self.create_recorded_callback(calls)
            step = self.create_recorded_step(calls)

            loop = DistributedTrainingLoop(
                MagicMock(DDP),
                step=step,
                rank=rank,
                device='cpu',
            )

            train_loader = self.create_dataloader(
                ('train1', 'train2', 'train3', 'train4'))
            val_loader = self.create_dataloader(('val1', 'val2', 'val3'))

            loop.fit(train_loader,
                     val_loader,
                     epochs=3,
                     callbacks=[callback,
                                EarlyStoppingAtSecondEpochEnd()])

            method_names = [name for name, _ in calls]
            if rank == loop._MAIN_PROCESS:
                assert method_names == [
                    'init_distributed',
                    'set_training_loop',
                    'on_training_begin',
                    # First epoch
                    'on_epoch_begin',
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    # 3 validation batches.
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    'on_epoch_end',
                    # Second epoch
                    'on_epoch_begin',
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    'on_train_batch_begin',
                    'train_step_distributed',
                    'on_train_batch_end',
                    # 3 validation batches.
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    'on_val_batch_begin',
                    'val_step_distributed',
                    'on_val_batch_end',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    'on_epoch_end',
                    # Early stopping occurs, thus end training.
                    'on_training_end',
                ]
            else:
                assert method_names == [
                    'init_distributed',
                    # First epoch
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    # 3 validation batches.
                    'val_step_distributed',
                    'val_step_distributed',
                    'val_step_distributed',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    # Second epoch
                    'reset_train_metrics_distributed',
                    'reset_val_metrics_distributed',
                    # 4 training batches.
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    'train_step_distributed',
                    # 3 validation batches.
                    'val_step_distributed',
                    'val_step_distributed',
                    'val_step_distributed',
                    # Compute an epoch's metrics.
                    'compute_train_metrics_synced',
                    'compute_val_metrics_synced',
                    # Early stopping occurs, thus end training.
                ]

    def test_three_epochs_with_early_stopping(self, world_size, backend,
                                              master_port):
        mp.spawn(self._test_three_epochs_with_early_stopping,
                 args=(master_port, world_size, backend),
                 nprocs=world_size)
