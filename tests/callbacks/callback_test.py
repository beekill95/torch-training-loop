from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from torch import nn
from training_loop import TrainingLoop
from training_loop.callbacks import Callback
from training_loop.exceptions import StopTraining


@pytest.fixture
def callback():
    return Callback()


def test_set_training_loop(callback: Callback):
    assert callback.training_loop is None
    assert callback.model is None

    model = nn.Sequential(nn.Linear(1, 1))
    loop = MagicMock(TrainingLoop)
    loop.model = model

    callback.set_training_loop(loop)

    assert callback.training_loop is loop
    assert callback.model is model


def test_on_training_begin(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_training_begin')
    callback.on('training_begin')
    spy.assert_called_once()


def test_on_training_end(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_training_end')
    callback.on('training_end')
    spy.assert_called_once()


def test_on_epoch_begin(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_epoch_begin')
    callback.on('epoch_begin', epoch=1)
    spy.assert_called_with(epoch=1)


def test_on_epoch_end(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_epoch_end')
    logs = {'epoch': 2}
    callback.on('epoch_end', epoch=2, logs=logs)
    spy.assert_called_with(epoch=2, logs=logs)


def test_on_train_batch_begin(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_train_batch_begin')
    callback.on('train_batch_begin', batch=1)
    spy.assert_called_once_with(batch=1)


def test_on_train_batch_end(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_train_batch_end')
    logs = {'batch': 2}
    callback.on('train_batch_end', batch=2, logs=logs)
    spy.assert_called_once_with(batch=2, logs=logs)


def test_on_val_batch_begin(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_val_batch_begin')
    callback.on('val_batch_begin', batch=3)
    spy.assert_called_once_with(batch=3)


def test_on_val_batch_end(mocker, callback: Callback):
    spy: MagicMock = mocker.spy(callback, 'on_val_batch_end')
    spy.side_effect = StopTraining()
    logs = {'batch': 4}
    callback.on('val_batch_end', batch=4, logs=logs)
    spy.assert_called_once_with(batch=4, logs=logs)


def test_raise_stop_training(mocker):

    class TestStopTrainingCallback(Callback):

        def on_training_begin(self):
            raise StopTraining()

        def on_training_end(self):
            raise StopTraining()

        def on_epoch_begin(self, epoch: int):
            raise StopTraining()

        def on_epoch_end(self, epoch: int, logs: dict[str, float]):
            raise StopTraining()

        def on_train_batch_begin(self, batch: int):
            raise StopTraining()

        def on_train_batch_end(self, batch: int, logs: dict[str, float]):
            raise StopTraining()

        def on_val_batch_begin(self, batch: int):
            raise StopTraining()

        def on_val_batch_end(self, batch: int, logs: dict[str, float]):
            raise StopTraining()

    # These events should not raise stop training.
    events = {
        'training_begin': {},
        'training_end': {},
        'epoch_begin': {
            'epoch': 1
        },
        'train_batch_begin': {
            'batch': 1
        },
        'train_batch_end': {
            'batch': 2,
            'logs': {
                'batch': 2
            }
        },
        'val_batch_begin': {
            'batch': 3
        },
        'val_batch_end': {
            'batch': 4,
            'logs': {
                'batch': 4
            }
        },
    }
    for event, kwargs in events.items():
        callback = TestStopTrainingCallback()
        spy: MagicMock = mocker.spy(callback, 'on')
        callback.on(event, **kwargs)
        assert spy.spy_exception is None

    # Only in `epoch_end` can the exception be raised.
    callback = TestStopTrainingCallback()
    spy: MagicMock = mocker.spy(callback, 'on')
    with pytest.raises(StopTraining):
        callback.on('epoch_end', epoch=2, logs={'epoch': 2})
