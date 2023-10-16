from __future__ import annotations

import numpy as np
import pytest
import torch
from torcheval.metrics import Metric
from training_loop.training_loops.simple_training_loop import (
    SimpleTrainingLoop,
    calc_loss,
    clone_metrics,
    compute_metrics,
    transfer_data,
    reset_metrics,
    update_metrics,
)
from unittest.mock import MagicMock, call, patch


class TestTransferData:

    def create_fake_tensor(self):
        return MagicMock(torch.Tensor)

    def test_transfer_single_tensor(self):
        tensor = self.create_fake_tensor()
        transfer_data(tensor, 'cuda')
        tensor.to.assert_called_once_with('cuda')

    def test_transfer_list_of_tensors(self):
        tensors = [
            self.create_fake_tensor(),
            self.create_fake_tensor(),
            self.create_fake_tensor(),
        ]
        transfer_data(tensors, 'cuda')
        for tensor in tensors:
            tensor.to.assert_called_once_with('cuda')

    def test_transfer_tuple_of_tensors(self):
        tensors = (
            self.create_fake_tensor(),
            self.create_fake_tensor(),
            self.create_fake_tensor(),
        )
        transfer_data(tensors, 'cuda')
        for tensor in tensors:
            tensor.to.assert_called_once_with('cuda')

    def test_transfer_dict_of_tensors(self):
        tensors = {
            'input1': self.create_fake_tensor(),
            'input2': self.create_fake_tensor(),
            'input3': self.create_fake_tensor(),
        }
        transfer_data(tensors, 'cuda')
        for tensor in tensors.values():
            tensor.to.assert_called_once_with('cuda')


class TestMetricsUtilities:

    def create_fake_metric(self):
        return MagicMock(Metric)

    # Test _clone_metrics()
    def test_clone_single_metric(self):
        metric = ('fake', self.create_fake_metric())
        with patch(
                'training_loop.training_loops.simple_training_loop.clone_metric'
        ) as clone_metric:
            clone_metric.return_value = metric[1]
            results = clone_metrics(metric)
            clone_metric.assert_called_once_with(metric[1])

            assert results == metric

    def test_clone_list_metrics(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
        ]

        with patch(
                'training_loop.training_loops.simple_training_loop.clone_metric'
        ) as clone_metric:
            clone_metric.side_effect = lambda x: x

            results = clone_metrics(metrics)
            clone_metric.assert_has_calls([
                call(metrics[0][1]),
                call(metrics[1][1]),
                call(metrics[2][1]),
            ],
                                          any_order=False)

            assert results == metrics

    def test_clone_dict_metrics(self):
        metrics = {
            'fake1': [
                ('fake1.1', self.create_fake_metric()),
                ('fake1.2', self.create_fake_metric()),
            ],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }

        with patch(
                'training_loop.training_loops.simple_training_loop.clone_metric'
        ) as clone_metric:
            clone_metric.side_effect = lambda x: x

            results = clone_metrics(metrics)
            clone_metric.assert_has_calls([
                call(metrics['fake1'][0][1]),
                call(metrics['fake1'][1][1]),
                call(metrics['fake2'][1]),
            ],
                                          any_order=False)

            assert results == metrics

    # Test _reset_metrics()
    def test_reset_single_metric(self):
        metric = self.create_fake_metric()
        reset_metrics(('fake_metric', metric))
        metric.reset.assert_called_once()

    def test_reset_list_of_metrics(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
            ('fake4', self.create_fake_metric()),
        ]
        reset_metrics(metrics)
        for _, metric in metrics:
            metric.reset.assert_called_once()

    def test_reset_dictionary_of_metrics(self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }
        reset_metrics(metrics)
        for _, metric in metrics['fake1']:
            metric.reset.assert_called_once()

        metrics['fake2'][1].reset.assert_called_once()

    # Test _update_metrics()
    def test_update_single_metric_multiple_outputs(self):
        metric = ('fake', self.create_fake_metric())
        y_pred = [torch.ones((1, )) * 2, torch.ones((1, )) * 3]
        y_true = [torch.ones((1, )) * 4, torch.ones((1, )) * 5]

        with pytest.raises(AssertionError):
            update_metrics(metric, y_pred=y_pred, y_true=y_true)

    def test_update_single_metric_dict_outputs(self):
        metric = ('fake', self.create_fake_metric())
        y_pred = {'out1': torch.ones((1, )) * 2, 'out2': torch.ones((1, )) * 3}
        y_true = {'out1': torch.ones((1, )) * 4, 'out2': torch.ones((1, )) * 5}

        with pytest.raises(AssertionError):
            update_metrics(metric, y_pred=y_pred, y_true=y_true)

    def test_update_single_metric_single_output(self):
        metric = ('fake', self.create_fake_metric())
        y_pred = torch.ones((1, )) * 2
        y_true = torch.ones((1, )) * 3

        update_metrics(metric, y_pred=y_pred, y_true=y_true)
        metric[1].update.assert_called_once_with(y_pred, y_true)

    def test_update_list_metrics_single_output(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
            ('fake4', self.create_fake_metric()),
        ]

        y_pred = torch.ones((1, )) * 2
        y_true = torch.ones((1, )) * 3
        update_metrics(metrics, y_pred=y_pred, y_true=y_true)

        for _, metric in metrics:
            metric.update.assert_called_once_with(y_pred, y_true)

    def test_update_list_metrics_sequence_outputs_different_length(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
            ('fake4', self.create_fake_metric()),
        ]

        y_pred = [torch.ones((1, )) * 2, torch.ones((1, )) * 3]
        y_true = [torch.ones((1, )) * 4, torch.ones((1, )) * 5]
        with pytest.raises(AssertionError):
            update_metrics(metrics, y_pred=y_pred, y_true=y_true)

    def test_update_list_metrics_list_outputs(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
            ('fake4', self.create_fake_metric()),
        ]

        y_pred = [
            torch.ones((1, )) * 2,
            torch.ones((1, )) * 3,
            torch.ones((1, )) * 4,
            torch.ones((1, )) * 5
        ]
        y_true = [
            torch.ones((1, )) * 6,
            torch.ones((1, )) * 7,
            torch.ones((1, )) * 8,
            torch.ones((1, )) * 9
        ]

        update_metrics(metrics, y_pred=y_pred, y_true=y_true)

        for (_, metric), yp, yt in zip(metrics, y_pred, y_true):
            metric.update.assert_called_once_with(yp, yt)

    def test_update_list_metrics_tuple_outputs(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
        ]

        y_pred = (
            torch.ones((1, )) * 2,
            torch.ones((1, )) * 3,
            torch.ones((1, )) * 4,
        )
        y_true = (
            torch.ones((1, )) * 5,
            torch.ones((1, )) * 6,
            torch.ones((1, )) * 7,
        )

        update_metrics(metrics, y_pred=y_pred, y_true=y_true)

        for (_, metric), yp, yt in zip(metrics, y_pred, y_true):
            metric.update.assert_called_once_with(yp, yt)

    def test_update_dict_metrics_single_output(self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }
        y_pred = torch.ones((1, )) * 2
        y_true = torch.ones((1, )) * 3

        with pytest.raises(AssertionError):
            update_metrics(metrics, y_pred=y_pred, y_true=y_true)

    def test_update_dict_metrics_list_outputs(self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }
        y_pred = [torch.ones((1, )) * 1, torch.ones((1, )) * 2]
        y_true = [torch.ones((1, )) * 3, torch.ones((1, )) * 4]

        with pytest.raises(AssertionError):
            update_metrics(metrics, y_pred=y_pred, y_true=y_true)

    def test_update_dict_metrics_dict_of_single_outputs(self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }
        y_pred = {
            'fake1': torch.ones((1, )) * 1,
            'fake2': torch.ones((1, )) * 2,
        }
        y_true = {
            'fake1': torch.ones((1, )) * 3,
            'fake2': torch.ones((1, )) * 4
        }

        update_metrics(metrics, y_pred=y_pred, y_true=y_true)

        metrics['fake1'][0][1].update.assert_called_once_with(
            y_pred['fake1'], y_true['fake1'])
        metrics['fake1'][1][1].update.assert_called_once_with(
            y_pred['fake1'], y_true['fake1'])
        metrics['fake2'][1].update.assert_called_once_with(
            y_pred['fake2'], y_true['fake2'])

    def test_update_dict_metrics_dict_of_sequence_outputs_different_length(
            self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }
        y_pred = {
            'fake1': [
                torch.ones((1, )) * 1,
                torch.ones((1, )) * 1.3,
                torch.ones((1, )) * 1.7,
            ],
            'fake2':
            torch.ones((1, )) * 2,
        }
        y_true = {
            'fake1': [
                torch.ones((1, )) * 3,
                torch.ones((1, )) * 3.3,
                torch.ones((1, )) * 3.7,
            ],
            'fake2':
            torch.ones((1, )) * 4
        }

        with pytest.raises(AssertionError):
            update_metrics(metrics, y_pred=y_pred, y_true=y_true)

    def test_update_dict_metrics_dict_of_sequence_outputs(self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }
        y_pred = {
            'fake1': [
                torch.ones((1, )) * 1,
                torch.ones((1, )) * 1.3,
            ],
            'fake2': torch.ones((1, )) * 2,
        }
        y_true = {
            'fake1': [
                torch.ones((1, )) * 3,
                torch.ones((1, )) * 3.3,
            ],
            'fake2': torch.ones((1, )) * 4
        }

        update_metrics(metrics, y_pred=y_pred, y_true=y_true)

        metrics['fake1'][0][1].update.assert_called_once_with(
            y_pred['fake1'][0], y_true['fake1'][0])
        metrics['fake1'][1][1].update.assert_called_once_with(
            y_pred['fake1'][1], y_true['fake1'][1])
        metrics['fake2'][1].update.assert_called_once_with(
            y_pred['fake2'], y_true['fake2'])

    # Test _compute_metrics()
    def test_compute_single_metric(self):
        metric = ('fake', self.create_fake_metric())
        metric[1].compute.return_value = torch.ones((1, )) * 0.58
        results = compute_metrics(metric)

        assert pytest.approx(results) == {'fake': 0.58}

    def test_compute_list_metrics(self):
        metrics = [
            ('fake1', self.create_fake_metric()),
            ('fake2', self.create_fake_metric()),
            ('fake3', self.create_fake_metric()),
            ('fake4', self.create_fake_metric()),
        ]
        metrics[0][1].compute.return_value = torch.ones((1, )) * 0.1
        metrics[1][1].compute.return_value = torch.ones((1, )) * 0.2
        metrics[2][1].compute.return_value = torch.ones((1, )) * 0.3
        metrics[3][1].compute.return_value = torch.ones((1, )) * 0.4

        results = compute_metrics(metrics)
        assert pytest.approx(results) == {
            'fake1': 0.1,
            'fake2': 0.2,
            'fake3': 0.3,
            'fake4': 0.4,
        }

    def test_compute_dict_metrics(self):
        metrics = {
            'fake1': [('fake1.1', self.create_fake_metric()),
                      ('fake1.2', self.create_fake_metric())],
            'fake2': ('fake2.1', self.create_fake_metric()),
        }

        metrics['fake1'][0][1].compute.return_value = torch.ones((1, )) * 0.1
        metrics['fake1'][1][1].compute.return_value = torch.ones((1, )) * 0.2
        metrics['fake2'][1].compute.return_value = torch.ones((1, )) * 0.3

        results = compute_metrics(metrics)
        assert pytest.approx(results) == {
            'fake1_fake1.1': 0.1,
            'fake1_fake1.2': 0.2,
            'fake2_fake2.1': 0.3,
        }
