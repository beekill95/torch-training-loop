from __future__ import annotations

from collections import ChainMap
import torch
from torch import nn, optim
from torcheval.metrics import Metric, Mean
from torcheval.metrics.toolkit import clone_metric
from typing import Callable, Iterator, Tuple, Sequence, Union, Dict, List

from .training_step import TrainingStep, TDevice

# Loss Functions.
TLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TLoss = Union[TLossFn, Sequence[TLossFn], Dict[str, Union[TLossFn,
                                                          Sequence[TLossFn]]]]
TLossWeights = Union[Sequence[float], Dict[str, Union[float, Sequence[float]]]]

# Metrics.
NamedMetric = Tuple[str, Metric[torch.Tensor]]
TMetrics = Union[NamedMetric, List[NamedMetric],
                 Dict[str, Union[NamedMetric, List[NamedMetric]]]]

# Optimizer function.
TOptimFn = Callable[[Iterator[nn.Parameter]], optim.Optimizer]

# Data passed in train and val steps.
TInputs = Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]
TOutputs = TInputs
TSampleWeights = TOutputs
TSimpleData = Union[Tuple[TInputs, TOutputs], Tuple[TInputs, TOutputs,
                                                    TSampleWeights]]


class SimpleTrainingStep(TrainingStep[nn.Module, TSimpleData]):
    """
    A simple training step that implements the base TrainingStep class.
    This training step, accepting an optimizer, loss functions, and metrics,
    allows training models that receive a single or multiple inputs and
    return a single or multiple outputs.

    In particular, this training step expects dataloaders to return a batch
    of data in a tuple of two or three elements. In the case of two elements,
    the first element is the input(s) and the second element is the expected output(s).
    In the case of three-element tuple, then the third element would be the sample
    weights given to each sample in the batch.

    Both the input(s), expected output(s), and the sample weights can be a single tensor
    (single-input/single-output models) or a sequence of tensors or a mapping between
    a name and a tensor (for multi-input/multi-output models). Another constraint is
    that the expected output(s) and the sample weights (if exists) must have the same
    data structure.
    """

    def __init__(
        self,
        *,
        optimizer_fn: TOptimFn,
        loss: TLoss,
        loss_weights: TLossWeights | None = None,
        metrics: TMetrics | None = None,
    ) -> None:
        """
        Construct a simple training step that works with single- or multi-output models.

        Parameters:
            optimizer_fn: A function that receives model's parameters and return an optimizer.

            loss: Loss functions. These loss functions should accept two positional
                parameters (`input` and `target`) and should return loss value for each
                element in a mini-batch (`reduction` is set to 'none').

                There are additional constraints on the data structure containing these
                loss functions:
                    * If `y_pred` (model's output) is a tensor, and consequently
                    `y_true` (returned by dataloaders) is a tensor as well,
                    loss must be a single function or a sequence of functions.
                    * If `y_pred` and `y_true` are sequence of tensors,
                    loss must be a sequence of functions of the same length.
                    * If `y_pred` and `y_true` are mappings, then loss must also
                    be a mapping. The corresponding values in each dictionary must
                    follow the above rules.

            loss_weights (optional): Weights given to each loss in the case multiple
                loss functions:
                    * If `loss_weights` is None, it means that each loss will have the
                    same weight = 1.
                    * If `loss` is a sequence of functions, then `loss_weights` must
                    be a sequence of weights of the same length.
                    * If `loss` is a dictionary, then `loss_weights` must also be
                    a dictionary.

            metrics: Metrics to monitor training/validation progress.

                Similar to the loss functions, these metrics must follow the
                constraints (see `update_metrics()`).
        """
        self.optim_fn = optimizer_fn
        self._loss_fn = loss
        self._loss_weights = loss_weights

        self._train_metrics = metrics
        self._val_metrics = clone_metrics(metrics)

    def init(self, model: nn.Module, device: TDevice) -> None:
        self._optim = self.optim_fn(model.parameters())

        self._train_loss = Mean(device=device)
        self._train_metrics = move_metrics_to_device(self._train_metrics,
                                                     device)

        self._val_loss = Mean(device=device)
        self._val_metrics = move_metrics_to_device(self._val_metrics, device)

    def train_step(self, model: nn.Module, data: TSimpleData,
                   device: TDevice) -> dict[str, float]:
        """
        Perform one training step on the given batch data.

        Parameters:
            data: Data returned by train dataloader. Data can be a tuple
            of two or three elements. For two-element tuples, the first
            element is the input data and the second element is the expected
            output data. For three-element tuples, the third element is
            sample weights for each sample in a mini-batch.
            The input, expected output, and sample weights can be a single tensor,
            (single-input single-output models) or a sequence of tensors,
            or a mapping between a str and a tensor (multi-input multi-output models).

        Returns:
            A dictionary containing metrics and loss values to show
            the training progress.
        """
        model = model.train()

        loss, y_pred, y_true = self._forward_pass(model, data, device)
        self._update_train_metrics(train_loss=loss,
                                   y_pred=y_pred,
                                   y_true=y_true)

        # Perform back-propagation.
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        # Return the metrics.
        return self.compute_train_metrics()

    @torch.no_grad()
    def val_step(self, model: nn.Module, data: TSimpleData,
                 device: TDevice) -> dict[str, float]:
        """
        Perform one validation step on the given batch data.

        Parameters:
            data: Similar to that of `train_step()`.

        Returns:
            A dictionary containing metrics and loss values to show the
            validation progress.
        """
        model = model.eval()

        loss, y_pred, y_true = self._forward_pass(model, data, device)
        self._update_val_metrics(val_loss=loss, y_pred=y_pred, y_true=y_true)

        # Return the metrics.
        return self.compute_val_metrics()

    def _forward_pass(
            self, model: nn.Module, data: TSimpleData,
            device: TDevice) -> Tuple[torch.Tensor, TOutputs, TOutputs]:
        """
        Perform a forward pass to obtain predicted values and loss.

        Parameters:
            model: nn.Module
            data: Data returned by a dataloader.
            device: Training device.

        Returns:
            A tuple of (loss, y_pred, y_true).
        """
        # Transfer data to the same device as model.
        data = self._transfer_to_device(data, device)

        # Unpack data.
        if len(data) == 2:
            X, y = data
            sample_weights = None
        else:
            X, y, sample_weights = data

        # Feed the model with input data.
        y_pred = model(X)

        # Calculate the loss and update metrics.
        loss = calc_loss(self._loss_fn,
                         y_pred=y_pred,
                         y_true=y,
                         loss_weights=self._loss_weights,
                         sample_weights=sample_weights)

        return loss, y_pred, y

    @torch.no_grad()
    def reset_train_metrics(self):
        self._train_loss.reset()
        reset_metrics(self._train_metrics)

    @torch.no_grad()
    def reset_val_metrics(self):
        self._val_loss.reset()
        reset_metrics(self._val_metrics)

    @torch.no_grad()
    def compute_train_metrics(self) -> dict[str, float]:
        return {
            'loss': self._train_loss.compute().detach().cpu().item(),
            **compute_metrics(self._train_metrics),
        }

    @torch.no_grad()
    def compute_val_metrics(self) -> dict[str, float]:
        return {
            'loss': self._val_loss.compute().detach().cpu().item(),
            **compute_metrics(self._val_metrics),
        }

    def _transfer_to_device(self, data: TSimpleData,
                            device: TDevice) -> TSimpleData:
        X = transfer_data(data[0], device)
        y = transfer_data(data[1], device)
        if len(data) == 3:
            sample_weights = transfer_data(data[2], device)
            return X, y, sample_weights

        return X, y

    @torch.no_grad()
    def _update_train_metrics(self, *, train_loss: torch.Tensor,
                              y_pred: torch.Tensor,
                              y_true: torch.Tensor) -> None:
        self._train_loss.update(train_loss)
        update_metrics(self._train_metrics, y_pred=y_pred, y_true=y_true)

    @torch.no_grad()
    def _update_val_metrics(self, *, val_loss: torch.Tensor,
                            y_pred: torch.Tensor,
                            y_true: torch.Tensor) -> None:
        self._val_loss.update(val_loss)
        update_metrics(self._val_metrics, y_pred=y_pred, y_true=y_true)


def transfer_data(data: TInputs, device: str | torch.device) -> TInputs:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Sequence):
        return [d.to(device) for d in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError(f'Unknown data structure: {data}.')


# Loss stuffs.
def _calc_single_loss(
    loss_fn: TLossFn,
    *,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply a loss function to the respective inputs and targets,
    taking into account the sample weights given to each sample.

    Parameters:
        loss_fn: A single loss function that accepts an input tensor
        and an output tensor. The function should return a tensor of
        a scalar, or a tensor of shape (batch_size, ) or same shape
        as the input.
        input: Input tensor.
        target: Target tensor.
        weight (optional): Sample weight given to each sample in the batch,
        can be a tensor of shape (batch_size, ) or having (or can be broadcasted to)
        the same shape as the input. If it is a scalar tensor, then the loss
        will simply be scaled by the given value.

    Returns: A scalar tensor.
    """
    assert isinstance(input, torch.Tensor) and isinstance(
        target, torch.Tensor) and (weight is None
                                   or isinstance(weight, torch.Tensor))

    # Loss functions in torch expect input first, and then target.
    loss = loss_fn(input, target)

    # `loss` can be a scalar tensor, a tensor of shape (batch_size, ) or
    # a tensor having the same shape as the inputs.
    # `weight` can be a tensor of shape (batch_size, ) or
    # a tensor having the same shape as the inputs.

    if weight is None:
        return torch.mean(loss)
    else:
        # The case of a scalar loss, or the dimensions of
        # both loss and weight matched.
        if loss.ndim == 0 or loss.ndim == weight.ndim:
            return torch.mean(loss * weight)

        # Otherwise, loss's shape is (batch_size, d0, ..., dN)
        # while weight's shape is (batch_size, )
        elif weight.ndim == 1 and weight.shape[0] == loss.shape[0]:
            # Append new dimensions to the *right* of weight to match
            # that of loss.
            weight = weight.view((-1, ) + (1, ) * (loss.ndim - 1))

            return torch.mean(loss * weight)

        raise ValueError(
            'Incomptible loss and sample weight shape. '
            f'Loss\' shape={loss.shape} while weight\'s shape={weight.shape}')


def _loss_weighted_average(
    losses: Sequence[float] | dict[str, float],
    weights: Sequence[float] | dict[str, float | Sequence[float]] | None,
):
    if isinstance(losses, Sequence):
        if weights is None:
            return sum(losses) / float(len(losses))
        elif isinstance(weights, Sequence):
            total_loss = 0.
            total_weight = 0.

            assert len(losses) == len(
                weights), 'Some loss functions\' weight(s) were not provided!'

            for loss, weight in zip(losses, weights):
                total_loss += loss * weight
                total_weight += weight

            return total_loss / total_weight
    else:
        if weights is None:
            return sum(losses.values()) / float(len(losses))
        elif isinstance(weights, dict):
            total_loss = 0.
            total_weight = 0.

            for key in losses.keys():
                w = weights[key]
                w = w if isinstance(w, float) else sum(w)
                total_loss += losses[key] * w
                total_weight += w

            return total_loss / total_weight

    raise ValueError('Incomptible type between losses and loss weights.\n'
                     f'Loss = {losses}\nLoss weights = {weights}.')


def calc_loss(
    loss_fns: TLoss,
    *,
    y_pred: TOutputs,
    y_true: TOutputs,
    sample_weights: TSampleWeights | None,
    loss_weights: TLossWeights | None,
) -> torch.Tensor:
    """
    Calculate loss function.

    Parameters:
        loss_fns: Loss functions. Each loss function should return a single-element
        tensor for each sample in the batch.
        loss_weights (optional): Weight given to each loss function.
        y_pred: Predicted outputs, must have the same data structure as `y_true`
        and `sample_weights` (if exists).
        y_true: True outputs, must have the same data structure as `y_pred`
        and `sample_weights` (if exists).
        sample_weights (optional): Weights given to each sample in the batch.

    Returns:
        A tensor of average loss.

    Note:
        * `y_true`, `y_pred`, and `sample_weights` must have the same data structure;
        * `y_true`, `y_pred`, and `sample_weights` must be compatible with the given
        loss functions:
            * If `y_true` and `y_pred` are a single tensor, then `loss_fn`
            can be a single function or a sequence of functions;
            * If `y_true` and `y_pred` are a sequence of tensors, then `loss_fn`
            has to be a sequence of functions with the same length;
            * If `y_true` and `y_pred` are a mapping, then `loss_fn` has
            to be a mapping. Corresponding values in each key must follow the
            rules above.
    """

    if isinstance(loss_fns, dict):
        assert isinstance(y_pred, dict) and isinstance(y_true, dict)
        assert (loss_weights is None) or isinstance(loss_weights, dict)
        assert (sample_weights is None) or isinstance(sample_weights, dict)

        losses = {
            key:
            calc_loss(
                loss_fns[key],
                y_pred=y_pred[key],
                y_true=y_true[key],
                sample_weights=sample_weights[key]
                if sample_weights is not None else None,
                loss_weights=loss_weights[key]
                if loss_weights is not None else None,
            )
            for key in loss_fns.keys()
        }
        return _loss_weighted_average(losses, loss_weights)

    elif isinstance(loss_fns, Sequence):
        if isinstance(y_pred, Sequence):
            assert len(loss_fns) == len(y_pred) == len(
                y_true
            ), 'The number of loss functions should match the number of outputs.'

            if sample_weights is None:
                sample_weights = (None, ) * len(loss_fns)

            losses = []
            for loss_fn, input, target, sample_weight in zip(
                    loss_fns, y_pred, y_true, sample_weights):
                losses.append(
                    _calc_single_loss(loss_fn,
                                      input=input,
                                      target=target,
                                      weight=sample_weight))

        elif isinstance(y_pred, torch.Tensor):
            # WONDERING: in this case, can the sample weights be a sequence?
            # If it is, then we can have different sample weights for each loss function.
            # Should we allow it?
            losses = [
                _calc_single_loss(loss_fn,
                                  input=y_pred,
                                  target=y_true,
                                  weight=sample_weights)
                for loss_fn in loss_fns
            ]
        else:
            raise ValueError(
                f'Unsupported output type for loss calculation: {y_pred}')

        return _loss_weighted_average(losses, loss_weights)

    else:
        return _calc_single_loss(loss_fns,
                                 input=y_pred,
                                 target=y_true,
                                 weight=sample_weights)


# Metrics-Related Functions.
def clone_metrics(metrics: TMetrics) -> TMetrics:
    if isinstance(metrics, list):
        return [(name, clone_metric(metric)) for name, metric in metrics]
    elif isinstance(metrics, dict):
        return {
            key: clone_metrics(submetrics)
            for key, submetrics in metrics.items()
        }
    else:
        name, metric = metrics
        return name, clone_metric(metric)


def move_metrics_to_device(metrics: TMetrics, device: TDevice) -> TMetrics:
    if isinstance(metrics, list):
        return [(name, metric.to(device)) for name, metric in metrics]
    elif isinstance(metrics, dict):
        return {
            key: move_metrics_to_device(submetrics, device)
            for key, submetrics in metrics.items()
        }
    else:
        name, metric = metrics
        return name, metric.to(device)


def reset_metrics(metrics: TMetrics) -> None:
    if isinstance(metrics, list):
        for _, metric in metrics:
            metric.reset()
    elif isinstance(metrics, dict):
        for _, submetrics in metrics.items():
            reset_metrics(submetrics)
    else:
        metrics[1].reset()


def update_metrics(
    metrics: TMetrics,
    *,
    y_pred: TOutputs,
    y_true: TOutputs,
) -> None:
    """
    Update metrics' internal state.

    Parameters:
        metrics: A metric, a list of metrics, or a mapping of metrics to be updated.
        y_pred: Predicted values.
        y_true: True values.

    Note:
        * `y_true` and `y_pred` must have the same data structure;
        * If `y_true` and `y_pred` are torch.Tensor,
        `metrics` must either be a single metric or a list of metrics;
        * If `y_true` and `y_pred` are sequences of tensors,
        `metrics` must be a list of metrics of the same length;
        * If `y_true` and `y_pred` are dictionaries,
        `metrics` must be a dictionary and each value in the dictionaries
        must follow the above rules.
    """
    if isinstance(metrics, dict):
        assert isinstance(y_true, dict) and isinstance(y_pred, dict)

        for key, submetrics in metrics.items():
            update_metrics(submetrics, y_true=y_true[key], y_pred=y_pred[key])
    elif isinstance(metrics, list):
        if isinstance(y_true, torch.Tensor):
            assert isinstance(y_pred, torch.Tensor)

            for _, metric in metrics:
                metric.update(y_pred, y_true)
        elif isinstance(y_true, Sequence):
            assert isinstance(y_pred, Sequence)
            assert len(metrics) == len(y_true) == len(y_pred)

            for (_, metric), target, pred in zip(metrics, y_true, y_pred):
                metric.update(pred, target)
        else:
            raise ValueError(
                'If `metrics` is a list, then both `y_true` and `y_pred`'
                ' must either be a single tensor or a sequence of tensors.')
    else:
        # `metrics` is just a single metric.
        assert isinstance(y_true, torch.Tensor) and isinstance(
            y_pred, torch.Tensor)
        metrics[1].update(y_pred, y_true)


def compute_metrics(metrics: TMetrics) -> dict[str, float]:

    def compute(metric: Metric[torch.Tensor]):
        return metric.compute().detach().cpu().item()

    if isinstance(metrics, list):
        return {name: compute(metric) for name, metric in metrics}
    elif isinstance(metrics, dict):
        results = [{
            f'{key}_{name}': value
            for name, value in compute_metrics(submetrics).items()
        } for key, submetrics in metrics.items()]

        return dict(ChainMap(*results))
    else:
        name, metric = metrics
        return {name: compute(metric)}
