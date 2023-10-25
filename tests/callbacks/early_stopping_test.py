from __future__ import annotations

import pytest
from torch import nn
from training_loop import TrainingLoop
from training_loop.callbacks import EarlyStopping
from training_loop.exceptions import StopTraining
from unittest.mock import Mock


def test_raise_StopTraining_when_monitored_value_not_decreasing():
    callback = EarlyStopping('loss', 'min', 5, False)
    losses = [1., 2., 3., 4., 5., 6., 7.]

    # Begin training.
    callback.on('training_begin')

    # Call the callback for some times to set up the state.
    for epoch, loss in enumerate(losses[:-1]):
        callback.on('epoch_end', epoch=epoch, logs={'loss': loss})

    # The callback should raise StopTraining if it waits long enough.
    with pytest.raises(StopTraining):
        callback.on('epoch_end',
                    epoch=len(losses) - 1,
                    logs={'loss': losses[-1]})


def test_raise_StopTraining_when_monitored_value_not_increasing():
    callback = EarlyStopping('f1', 'max', 5, False)
    f1s = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    # Begin training.
    callback.on('training_begin')

    # Call the callback for some times to set up the state.
    for epoch, f1 in enumerate(f1s[:-1]):
        callback.on('epoch_end', epoch=epoch, logs={'f1': f1})

    # The callback should raise StopTraining if it waits long enough.
    with pytest.raises(StopTraining):
        callback.on('epoch_end', epoch=len(f1s) - 1, logs={'f1': f1s[-1]})


def test_restore_best_weights():
    fake_model = Mock(nn.Module)
    fake_loop = Mock(TrainingLoop)
    fake_loop.model = fake_model

    state_dict = {'best_weights': (1, 2, 3)}
    fake_model.state_dict.return_value = state_dict

    # Init callback.
    callback = EarlyStopping('f1', 'max', 5, True)
    callback.set_training_loop(fake_loop)
    callback.on('training_begin')

    # Let the callback store the model's state dict.
    callback.on('epoch_end', epoch=1, logs={'f1': 1.0})

    # End the training, the best weights should be restored,
    # even if it is not due to early stopping.
    callback.on('training_end')
    fake_model.load_state_dict.assert_called_once_with(state_dict)
