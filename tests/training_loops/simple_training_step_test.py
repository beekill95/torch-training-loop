from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torcheval.metrics import Metric
from training_loop.training_loops.simple_training_step import SimpleTrainingStep
from unittest.mock import MagicMock, call, patch


class TrainingStepInstances:

    def __init__(self) -> None:
        self.model = MagicMock(nn.Module)
        self.model.train.return_value = self.model
        self.model.eval.return_value = self.model

        self.optim = MagicMock(optim.Adam)
        self.optim.return_value = self.optim

        self.loss_fn = {'f1': MagicMock(nn.L1Loss)}
        self.loss_weights = {'f1': 0.1}

        self.metrics = ('metric', MagicMock(Metric))
        self.device = MagicMock(torch.device)

    def create_step(self):

        return SimpleTrainingStep(
            optimizer_fn=lambda params: self.optim(params),
            loss=self.loss_fn,
            loss_weights=self.loss_weights,
            metrics=self.metrics,
        )


@patch(
    'training_loop.training_loops.simple_training_step.move_metrics_to_device')
@patch('training_loop.training_loops.simple_training_step.clone_metrics')
def test_loop_init(clone_metrics, move_metrics):
    instances = TrainingStepInstances()
    parameters = (1, 2, 3)
    instances.model.parameters.return_value = parameters

    fake_metrics = ('fake_return', MagicMock(Metric))
    clone_metrics.return_value = fake_metrics
    step = instances.create_step()

    # Metrics are cloned.
    clone_metrics.assert_called_once_with(instances.metrics)

    with patch('training_loop.training_loops.simple_training_step.Mean'):
        step.init(instances.model, instances.device)

    # Optimizer initialized.
    instances.optim.assert_called_once_with(parameters)

    # Metrics are moved to the correct device.
    move_metrics.assert_has_calls([
        call(instances.metrics, instances.device),
        call(fake_metrics, instances.device),
    ],
                                  any_order=False)


class TestTraining:

    train_data = (
        torch.tensor(np.asarray([
            [1., 2., 3.],
            [2., 3., 4.],
        ])),
        torch.tensor(np.asarray([1., 0.])),
    )

    @patch('training_loop.training_loops.simple_training_step.Mean')
    @patch(
        'training_loop.training_loops.simple_training_step.move_metrics_to_device'
    )
    @patch('training_loop.training_loops.simple_training_step.compute_metrics')
    @patch('training_loop.training_loops.simple_training_step.update_metrics')
    @patch('training_loop.training_loops.simple_training_step.calc_loss')
    @patch('training_loop.training_loops.simple_training_step.transfer_data')
    def test_train_step(
        self,
        transfer_data,
        calc_loss,
        update_metrics,
        compute_metrics,
        move_metrics,
        metric_mean,
    ):
        # Set up instances.
        instances = TrainingStepInstances()
        y_pred = torch.tensor(np.asarray([0.9, 0.7]))
        instances.model.return_value = y_pred

        loss = MagicMock(torch.Tensor)
        loss.detach.return_value.cpu.return_value.item.return_value = 7.0
        calc_loss.return_value = loss

        metric_mean.return_value = metric_mean
        metric_mean.compute.return_value.detach.return_value.cpu.return_value.item.return_value = 7.0

        compute_metrics.return_value = {'f1': 0.2}

        transfer_data.side_effect = lambda x, _: x
        move_metrics.side_effect = lambda x, _: x

        # Call train step.
        step = instances.create_step()
        step.init(instances.model, instances.device)

        logs = step.train_step(instances.model, self.train_data,
                               instances.device)

        # Switch model to train mode.
        instances.model.train.assert_called_once()

        # Transfer data to device.
        transfer_data.assert_has_calls(
            [call(d, instances.device) for d in self.train_data])

        # Model was called with X.
        instances.model.assert_called_once_with(self.train_data[0])

        # Calculate loss function was called with correct parameters.
        calc_loss.assert_called_once_with(
            instances.loss_fn,
            y_pred=y_pred,
            y_true=self.train_data[1],
            sample_weights=None,
            loss_weights=instances.loss_weights,
        )

        # Update metrics function was called with correct parameters.
        metric_mean.update.assert_called_once_with(loss)
        update_metrics.assert_called_once_with(
            instances.metrics,
            y_pred=y_pred,
            y_true=self.train_data[1],
        )

        # Reset optimizer, loss backward and step optimizer.
        instances.optim.zero_grad.assert_called_once()
        loss.backward.assert_called_once()
        instances.optim.step.assert_called_once()

        # Compute train metrics and return.
        compute_metrics.assert_called_once_with(instances.metrics)
        assert logs == {'loss': 7.0, 'f1': 0.2}


class TestValidation:
    val_data = (
        torch.tensor(np.asarray([
            [1., 2., 3.],
            [2., 3., 4.],
        ])),
        torch.tensor(np.asarray([1., 0.])),
    )

    val_metrics = MagicMock(Metric)

    @patch('training_loop.training_loops.simple_training_step.Mean')
    @patch(
        'training_loop.training_loops.simple_training_step.move_metrics_to_device'
    )
    @patch('training_loop.training_loops.simple_training_step.compute_metrics')
    @patch('training_loop.training_loops.simple_training_step.update_metrics')
    @patch('training_loop.training_loops.simple_training_step.calc_loss')
    @patch('training_loop.training_loops.simple_training_step.transfer_data')
    @patch('training_loop.training_loops.simple_training_step.clone_metrics')
    def test_val_step(
        self,
        clone_metrics,
        transfer_data,
        calc_loss,
        update_metrics,
        compute_metrics,
        move_metrics,
        metric_mean,
    ):
        # Set up instances.
        instances = TrainingStepInstances()
        y_pred = torch.tensor(np.asarray([0.9, 0.7]))
        instances.model.return_value = y_pred

        loss = MagicMock(torch.Tensor)
        loss.detach.return_value.cpu.return_value.item.return_value = 7.0
        calc_loss.return_value = loss

        metric_mean.return_value = metric_mean
        metric_mean.compute.return_value.detach.return_value.cpu.return_value.item.return_value = 7.0

        compute_metrics.return_value = {'f1': 0.2}

        clone_metrics.return_value = (instances.metrics[0], self.val_metrics)

        transfer_data.side_effect = lambda x, _: x
        move_metrics.side_effect = lambda x, _: x

        # Call train step.
        step = instances.create_step()
        step.init(instances.model, instances.device)
        logs = step.val_step(instances.model, self.val_data, instances.device)

        # Switch model to eval mode.
        instances.model.eval.assert_called_once()

        # Transfer data to device.
        transfer_data.assert_has_calls(
            [call(d, instances.device) for d in self.val_data])

        # Model was called with X.
        instances.model.assert_called_once_with(self.val_data[0])

        # Calculate loss function was called with correct parameters.
        calc_loss.assert_called_once_with(
            instances.loss_fn,
            y_pred=y_pred,
            y_true=self.val_data[1],
            sample_weights=None,
            loss_weights=instances.loss_weights,
        )

        # Update metrics function was called with correct parameters.
        metric_mean.update.assert_called_once_with(loss)
        update_metrics.assert_called_once_with(
            (instances.metrics[0], self.val_metrics),
            y_pred=y_pred,
            y_true=self.val_data[1],
        )

        # Reset optimizer, loss backward and step optimizer.
        instances.optim.zero_grad.assert_not_called()
        loss.backward.assert_not_called()
        instances.optim.step.assert_not_called()

        # Compute train metrics and return.
        compute_metrics.assert_called_once_with(
            (instances.metrics[0], self.val_metrics))
        assert logs == {'loss': 7.0, 'f1': 0.2}
