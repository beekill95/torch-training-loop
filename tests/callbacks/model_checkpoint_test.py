from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from torch import nn
from training_loop import TrainingLoop
from training_loop.callbacks import ModelCheckpoint


@pytest.fixture
def fake_model():
    return Mock(nn.Module)


@pytest.fixture
def fake_loop(fake_model):
    loop = Mock(TrainingLoop)
    loop.model = fake_model
    return loop


def test_model_is_saved_at_every_epoch(fake_model, fake_loop):
    callback = ModelCheckpoint(
        "model.pth",
        save_weights_only=False,
        save_best_only=False,
    )
    callback.set_training_loop(fake_loop)
    callback.on("training_begin")

    for epoch in range(10):
        with patch("torch.save") as torch_save:
            callback.on("epoch_end", epoch=epoch, logs={})
            torch_save.assert_called_once_with(fake_model, "model.pth")


def test_model_weight_is_saved_at_every_epoch(fake_model, fake_loop):
    fake_state_dict = {"weight": (1, 2, 3)}
    fake_model.state_dict.return_value = fake_state_dict

    callback = ModelCheckpoint(
        "model.pth",
        save_weights_only=True,
        save_best_only=False,
    )
    callback.set_training_loop(fake_loop)
    callback.on("training_begin")

    for epoch in range(10):
        with patch("torch.save") as torch_save:
            callback.on("epoch_end", epoch=epoch, logs={})
            torch_save.assert_called_once_with(fake_state_dict, "model.pth")


def test_raise_error_when_monitored_value_not_exist():
    callback = ModelCheckpoint("model.pth", monitor="f1", mode="max")
    callback.on("training_begin")

    with pytest.raises(ValueError):
        callback.on("epoch_end", epoch=1, logs={"loss": 0.48})


def test_model_is_saved_when_performance_increase(fake_model, fake_loop):
    callback = ModelCheckpoint(
        "model.pth",
        save_weights_only=False,
        save_best_only=True,
        monitor="loss",
        mode="min",
    )
    callback.set_training_loop(fake_loop)
    callback.on("training_begin")

    # First time will always trigger a save.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=1, logs={"loss": 1.0})
        torch_save.assert_called_once_with(fake_model, "model.pth")

    # If the monitored value decreases, the model will be saved.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=2, logs={"loss": 0.9})
        torch_save.assert_called_once_with(fake_model, "model.pth")

    # If the monitored value increases, the model wont be saved.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=3, logs={"loss": 0.95})
        torch_save.assert_not_called()

    # If the monitored value decreases again, the model will be saved.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=3, logs={"loss": 0.85})
        torch_save.assert_called_once_with(fake_model, "model.pth")


def test_model_weight_is_saved_when_performance_decrease(fake_model, fake_loop):
    fake_state_dict = {"weight": (1, 2, 3)}
    fake_model.state_dict.return_value = fake_state_dict

    callback = ModelCheckpoint(
        "model.pth",
        save_weights_only=True,
        save_best_only=True,
        monitor="f1",
        mode="max",
    )
    callback.set_training_loop(fake_loop)
    callback.on("training_begin")

    # First time will always trigger a save.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=1, logs={"f1": 0.1})
        torch_save.assert_called_once_with(fake_state_dict, "model.pth")

    # If the monitored value decreases, the model will be saved.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=2, logs={"f1": 0.2})
        torch_save.assert_called_once_with(fake_state_dict, "model.pth")

    # If the monitored value increases, the model wont be saved.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=3, logs={"f1": 0.15})
        torch_save.assert_not_called()

    # If the monitored value decreases again, the model will be saved.
    with patch("torch.save") as torch_save:
        callback.on("epoch_end", epoch=3, logs={"f1": 0.25})
        torch_save.assert_called_once_with(fake_state_dict, "model.pth")


def test_save_model_at_every_epoch_with_callable_filename(fake_model, fake_loop):
    callback = ModelCheckpoint(
        lambda epoch, logs: Path(f'model_{epoch}_{logs["f1"]}.pth'),
        save_weights_only=False,
        save_best_only=False,
    )
    callback.set_training_loop(fake_loop)
    callback.on("training_begin")

    for epoch in range(10):
        f1 = epoch * 0.1 + 0.05
        with patch("torch.save") as torch_save:
            callback.on("epoch_end", epoch=epoch, logs={"f1": f1})
            torch_save.assert_called_once_with(
                fake_model,
                Path(f"model_{epoch}_{f1}.pth"),
            )
