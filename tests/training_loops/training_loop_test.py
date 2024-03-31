from __future__ import annotations

from unittest.mock import call
from unittest.mock import DEFAULT
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from torch import nn
from torch.utils.data import DataLoader
from training_loop import TrainingLoop
from training_loop import TrainingStep
from training_loop.callbacks import Callback
from training_loop.exceptions import StopTraining

from .utils import assert_dataframes_equal


def new_training_step_mock():
    return MagicMock(spec=TrainingStep)


def new_callback_mock():
    mock = MagicMock(spec=Callback)
    mock.on = lambda *args, **kwargs: Callback.on(mock, *args, **kwargs)

    return mock


@pytest.fixture
def fake_model():
    model = MagicMock(nn.Module)
    model.to.return_value = model

    return model


@pytest.fixture
def fake_callback():
    return new_callback_mock()


def test_init_loop_with_cpu_device(fake_model):
    loop = TrainingLoop(fake_model, step=new_training_step_mock(), device="cpu")
    fake_model.to.assert_called_once_with("cpu")
    assert loop.device == "cpu"
    assert loop.model is fake_model


def test_init_loop_with_cuda_device(fake_model):
    loop = TrainingLoop(fake_model, step=new_training_step_mock(), device="cuda")
    fake_model.to.assert_called_once_with("cuda")
    assert loop.device == "cuda"
    assert loop.model is fake_model


class TestTrainingLoopFit:
    train_data = ("train_1", "train_2", "train_3")
    val_data = ("val_1", "val_2", "val_3")
    train_step_return_values = (
        {"f1": 0.5, "batch": 1},
        {"f1": 0.6, "batch": 2},
        {"f1": 0.7, "batch": 3},
    )
    val_step_return_values = (
        {"f1": 0.3, "batch": 1},
        {"f1": 0.4, "batch": 2},
        {"f1": 0.5, "batch": 3},
    )

    @pytest.fixture
    def train_dataloader(self):
        dataloader = MagicMock(DataLoader)
        dataloader.__iter__.return_value = self.train_data
        dataloader.__len__.return_value = len(self.train_data)

        return dataloader

    @pytest.fixture
    def val_dataloader(self):
        dataloader = MagicMock(DataLoader)
        dataloader.__iter__.return_value = self.val_data
        dataloader.__len__.return_value = len(self.val_data)

        return dataloader

    @pytest.fixture
    def step(self):
        return new_training_step_mock()

    @pytest.fixture
    def loop(self, fake_model, step):
        empty_loop = TrainingLoop(fake_model, step=step)
        step.train_step.side_effect = self.train_step_return_values
        step.val_step.side_effect = self.val_step_return_values

        step.compute_train_metrics.return_value = {"f1": 0.8, "epoch": 1}
        step.compute_val_metrics.return_value = {"f1": 0.6, "epoch": 1}

        return empty_loop

    def test_calls_made_with_correct_arguments(
        self,
        train_dataloader,
        val_dataloader,
        loop,
        fake_callback,
        fake_model,
        step,
    ):
        loop.fit(
            train_dataloader,
            val_dataloader,
            epochs=1,
            callbacks=[fake_callback],
        )

        # Step init.
        step.init.assert_called_once_with(fake_model, "cpu")

        # Callbacks init.
        fake_callback.set_training_loop.assert_called_once_with(loop)

        # Assert callbacks' events.
        fake_callback.on_training_begin.assert_called_once()
        fake_callback.on_epoch_begin.assert_called_once_with(epoch=1)

        assert fake_callback.on_train_batch_begin.call_args_list == [
            call(batch=1),
            call(batch=2),
            call(batch=3),
        ]
        assert fake_callback.on_train_batch_end.call_args_list == [
            call(batch=1, logs={"f1": 0.5, "batch": 1}),
            call(batch=2, logs={"f1": 0.6, "batch": 2}),
            call(batch=3, logs={"f1": 0.7, "batch": 3}),
        ]

        assert fake_callback.on_val_batch_begin.call_args_list == [
            call(batch=1),
            call(batch=2),
            call(batch=3),
        ]
        assert fake_callback.on_val_batch_end.call_args_list == [
            call(batch=1, logs={"val_f1": 0.3, "val_batch": 1}),
            call(batch=2, logs={"val_f1": 0.4, "val_batch": 2}),
            call(batch=3, logs={"val_f1": 0.5, "val_batch": 3}),
        ]

        fake_callback.on_epoch_end.assert_called_once_with(
            epoch=1,
            logs={"f1": 0.8, "epoch": 1, "val_f1": 0.6, "val_epoch": 1},
        )
        fake_callback.on_training_end.assert_called_once()

        # Metrics stuffs.
        step.reset_train_metrics.assert_called_once()
        step.reset_val_metrics.assert_called_once()
        step.compute_train_metrics.assert_called_once()
        step.compute_val_metrics.assert_called_once()

    def test_fit_multiple_epochs(
        self,
        train_dataloader,
        val_dataloader,
        loop,
        fake_callback,
        step,
    ):

        def reset_mocks_return_values(*args, **kwargs):
            train_dataloader.__iter__.return_value = self.train_data
            val_dataloader.__iter__.return_value = self.val_data

            step.train_step.side_effect = self.train_step_return_values
            step.val_step.side_effect = self.val_step_return_values

            return DEFAULT

        fake_callback.on_epoch_end.side_effect = reset_mocks_return_values

        loop.fit(
            train_dataloader,
            val_dataloader,
            epochs=3,
            callbacks=[fake_callback],
        )

        # Callbacks init.
        fake_callback.set_training_loop.assert_called_once_with(loop)

        # Assert callbacks' events.
        fake_callback.on_training_begin.assert_called_once()
        assert fake_callback.on_epoch_begin.call_args_list == [
            call(epoch=1),
            call(epoch=2),
            call(epoch=3),
        ]

        assert fake_callback.on_train_batch_begin.call_count == 9
        assert fake_callback.on_train_batch_end.call_count == 9

        assert fake_callback.on_val_batch_begin.call_count == 9
        assert fake_callback.on_val_batch_end.call_count == 9

        assert fake_callback.on_epoch_end.call_args_list == [
            call(epoch=1, logs={"f1": 0.8, "epoch": 1, "val_f1": 0.6, "val_epoch": 1}),
            call(epoch=2, logs={"f1": 0.8, "epoch": 1, "val_f1": 0.6, "val_epoch": 1}),
            call(epoch=3, logs={"f1": 0.8, "epoch": 1, "val_f1": 0.6, "val_epoch": 1}),
        ]

        fake_callback.on_training_end.assert_called_once()

        # Metrics stuffs.
        assert step.reset_train_metrics.call_count == 3
        assert step.reset_val_metrics.call_count == 3
        assert step.compute_train_metrics.call_count == 3
        assert step.compute_val_metrics.call_count == 3

    def test_returned_histories(
        self,
        train_dataloader,
        val_dataloader,
        loop,
        fake_callback,
    ):
        train_history, val_history = loop.fit(
            train_dataloader,
            val_dataloader,
            epochs=1,
            callbacks=[fake_callback],
        )

        assert_dataframes_equal(
            train_history,
            pd.DataFrame(
                [
                    {"epoch": 1, "batch": 1, "f1": 0.5},
                    {
                        "epoch": 1,
                        "batch": 2,
                        "f1": 0.6,
                    },
                    {
                        "epoch": 1,
                        "batch": 3,
                        "f1": 0.7,
                    },
                    {
                        "epoch": 1,
                        "batch": -1,
                        "f1": 0.8,
                    },
                ]
            ).set_index(["epoch", "batch"], drop=False),
        )

        assert_dataframes_equal(
            val_history,
            pd.DataFrame(
                [
                    {"val_epoch": 1, "val_batch": 1, "val_f1": 0.3},
                    {
                        "val_epoch": 1,
                        "val_batch": 2,
                        "val_f1": 0.4,
                    },
                    {
                        "val_epoch": 1,
                        "val_batch": 3,
                        "val_f1": 0.5,
                    },
                    {
                        "val_epoch": 1,
                        "val_batch": -1,
                        "val_f1": 0.6,
                    },
                ]
            ).set_index(["val_epoch", "val_batch"], drop=False),
        )

    def test_early_stopping(
        self,
        train_dataloader,
        val_dataloader,
        loop,
        fake_callback,
        step,
    ):

        def reset_mocks_return_values(epoch, **kwargs):
            train_dataloader.__iter__.return_value = self.train_data
            val_dataloader.__iter__.return_value = self.val_data

            step.train_step.side_effect = self.train_step_return_values
            step.val_step.side_effect = self.val_step_return_values

            return DEFAULT

        fake_callback.on_epoch_begin.side_effect = reset_mocks_return_values
        fake_callback.on_epoch_end.side_effect = StopTraining()

        loop.fit(
            train_dataloader,
            val_dataloader,
            epochs=5,
            callbacks=[fake_callback],
        )

        # Assert callbacks' events.
        fake_callback.on_training_begin.assert_called_once()
        assert fake_callback.on_epoch_begin.call_args_list == [
            call(epoch=1),
        ]

        assert fake_callback.on_train_batch_begin.call_count == 3
        assert fake_callback.on_train_batch_end.call_count == 3

        assert fake_callback.on_val_batch_begin.call_count == 3
        assert fake_callback.on_val_batch_end.call_count == 3

        assert fake_callback.on_epoch_end.call_args_list == [
            call(epoch=1, logs={"f1": 0.8, "epoch": 1, "val_f1": 0.6, "val_epoch": 1}),
        ]

        fake_callback.on_training_end.assert_called_once()

        # Metrics stuffs.
        assert step.reset_train_metrics.call_count == 1
        assert step.reset_val_metrics.call_count == 1
        assert step.compute_train_metrics.call_count == 1
        assert step.compute_val_metrics.call_count == 1

    @pytest.mark.parametrize("verbose", [0, 1, 2, 5, 10])
    @patch("training_loop.training_loops.training_loop.ProgressReporter")
    def test_progress_reporter(
        self,
        reporter,
        train_dataloader,
        val_dataloader,
        loop,
        verbose,
    ):
        reporter_ctx = MagicMock()
        reporter.return_value.__enter__.return_value = reporter_ctx
        loop.fit(train_dataloader, val_dataloader, epochs=1, verbose=verbose)

        reporter.assert_called_once_with(
            1,
            total_epochs=1,
            total_batches=6,
            verbose=verbose,
        )

        assert reporter_ctx.next_batch.call_count == 6
        reporter_ctx.report_batch_progress.assert_has_calls(
            [
                call(
                    "Training",
                    {
                        "f1": 0.5,
                        "batch": 1,
                    },
                ),
                call(
                    "Training",
                    {
                        "f1": 0.6,
                        "batch": 2,
                    },
                ),
                call(
                    "Training",
                    {
                        "f1": 0.7,
                        "batch": 3,
                    },
                ),
                call(
                    "Validating",
                    {
                        "val_f1": 0.3,
                        "val_batch": 1,
                    },
                ),
                call(
                    "Validating",
                    {
                        "val_f1": 0.4,
                        "val_batch": 2,
                    },
                ),
                call(
                    "Validating",
                    {
                        "val_f1": 0.5,
                        "val_batch": 3,
                    },
                ),
            ],
            any_order=False,
        )
        reporter_ctx.report_epoch_progress.assert_called_once_with(
            "Finished",
            {
                "f1": 0.8,
                "epoch": 1,
                "val_f1": 0.6,
                "val_epoch": 1,
            },
        )


class TestTrainingLoopFitCallsOrder:

    def _record_call(
        self,
        method: str,
        call_orders: list[tuple[str, tuple[tuple, dict]]],
    ):

        def record(*args, **kwargs):
            call_orders.append((method, (args, kwargs)))
            return DEFAULT

        return record

    def recorded_fake_callback(
        self,
        fake_callback,
        call_orders: list[tuple[str, tuple[tuple, dict]]],
    ):
        fake_callback.on_training_begin.side_effect = self._record_call(
            "on_training_begin", call_orders
        )
        fake_callback.on_training_end.side_effect = self._record_call(
            "on_training_end", call_orders
        )

        fake_callback.on_train_batch_begin.side_effect = self._record_call(
            "on_train_batch_begin", call_orders
        )
        fake_callback.on_train_batch_end.side_effect = self._record_call(
            "on_train_batch_end", call_orders
        )

        fake_callback.on_val_batch_begin.side_effect = self._record_call(
            "on_val_batch_begin", call_orders
        )
        fake_callback.on_val_batch_end.side_effect = self._record_call(
            "on_val_batch_end", call_orders
        )

        fake_callback.on_epoch_begin.side_effect = self._record_call(
            "on_epoch_begin", call_orders
        )
        fake_callback.on_epoch_end.side_effect = self._record_call(
            "on_epoch_end", call_orders
        )

        return fake_callback

    def recorded_loop(self, fake_model, call_orders: list):
        step = new_training_step_mock()
        loop = TrainingLoop(fake_model, step)

        step.init.side_effect = self._record_call("init", call_orders)
        step.train_step.side_effect = self._record_call("train_step", call_orders)
        step.train_step.return_value = {"f1": 0.8}
        step.val_step.side_effect = self._record_call("val_step", call_orders)
        step.val_step.return_value = {"f1": 0.7}

        step.compute_train_metrics.side_effect = self._record_call(
            "compute_train_metrics", call_orders
        )
        step.compute_train_metrics.return_value = {"f1": 0.8}
        step.compute_val_metrics.side_effect = self._record_call(
            "compute_val_metrics", call_orders
        )
        step.compute_val_metrics.return_value = {"f1": 0.7}

        step.reset_train_metrics.side_effect = self._record_call(
            "reset_train_metrics", call_orders
        )
        step.reset_val_metrics.side_effect = self._record_call(
            "reset_val_metrics", call_orders
        )

        return loop

    def create_fake_dataloader(self, return_values):
        dataloader = MagicMock(DataLoader)
        dataloader.__iter__.return_value = iter(return_values)

        return dataloader

    def test_one_epoch(self, fake_model, fake_callback):
        call_orders: list[tuple[str, tuple[tuple, dict]]] = []
        callback = self.recorded_fake_callback(fake_callback, call_orders)
        loop = self.recorded_loop(fake_model, call_orders)

        train_dataloader = self.create_fake_dataloader(
            ("train1", "train2", "train3", "train4")
        )
        val_dataloader = self.create_fake_dataloader(("val1", "val2", "val3"))

        loop.fit(train_dataloader, val_dataloader, epochs=1, callbacks=[callback])

        methods = [name for name, _ in call_orders]
        assert methods == [
            "init",
            "on_training_begin",
            # First epoch
            "on_epoch_begin",
            "reset_train_metrics",
            "reset_val_metrics",
            # 4 training batches.
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            # 3 validation batches.
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            # Compute an epoch's metrics.
            "compute_train_metrics",
            "compute_val_metrics",
            "on_epoch_end",
            # End training.
            "on_training_end",
        ]

    def test_three_epochs(self, fake_model, fake_callback):

        class ResetDataCallback(Callback):

            def __init__(self, train_dataloader, val_dataloader, train_data, val_data):
                super().__init__()

                self.train_dataloader = train_dataloader
                self.val_dataloader = val_dataloader
                self.train_data = train_data
                self.val_data = val_data

            def on_epoch_begin(self, epoch: int):
                self.train_dataloader.__iter__.return_value = self.train_data
                self.val_dataloader.__iter__.return_value = self.val_data

        call_orders: list[tuple[str, tuple[tuple, dict]]] = []
        callback = self.recorded_fake_callback(fake_callback, call_orders)
        loop = self.recorded_loop(fake_model, call_orders)

        train_dataloader = MagicMock(DataLoader)
        val_dataloader = MagicMock(DataLoader)

        loop.fit(
            train_dataloader,
            val_dataloader,
            epochs=3,
            callbacks=[
                callback,
                ResetDataCallback(
                    train_dataloader,
                    val_dataloader,
                    ("train1", "train2", "train3", "train4"),
                    ("val1", "val2", "val3"),
                ),
            ],
        )

        methods = [name for name, _ in call_orders]
        assert methods == [
            "init",
            "on_training_begin",
            # First epoch
            "on_epoch_begin",
            "reset_train_metrics",
            "reset_val_metrics",
            # 4 training batches.
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            # 3 validation batches.
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            # Compute an epoch's metrics.
            "compute_train_metrics",
            "compute_val_metrics",
            "on_epoch_end",
            # End first epoch.
            # Second epoch
            "on_epoch_begin",
            "reset_train_metrics",
            "reset_val_metrics",
            # 4 training batches.
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            # 3 validation batches.
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            # Compute an epoch's metrics.
            "compute_train_metrics",
            "compute_val_metrics",
            "on_epoch_end",
            # End second epoch.
            # Third epoch
            "on_epoch_begin",
            "reset_train_metrics",
            "reset_val_metrics",
            # 4 training batches.
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            "on_train_batch_begin",
            "train_step",
            "on_train_batch_end",
            # 3 validation batches.
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            "on_val_batch_begin",
            "val_step",
            "on_val_batch_end",
            # Compute an epoch's metrics.
            "compute_train_metrics",
            "compute_val_metrics",
            "on_epoch_end",
            # End third epoch.
            # End training.
            "on_training_end",
        ]
