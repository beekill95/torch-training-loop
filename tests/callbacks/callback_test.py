from unittest.mock import MagicMock
import pytest
from training_loop.callbacks import Callback


def test_on_training_begin(mocker):
    callback = Callback()
    spy: MagicMock = mocker.spy(callback, 'on_training_begin')
    callback.on('training_begin')

    spy.assert_called_once()
