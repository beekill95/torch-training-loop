from __future__ import annotations

import abc
import torch
from typing import Union, Generic

from ..types import TData, TModel

TDevice = Union[torch.device, str]


class TrainingStep(Generic[TModel, TData], abc.ABC):

    @abc.abstractmethod
    def init(self, model: TModel, device: TDevice) -> None:
        """Initialize the instance to be ready for training."""
        pass

    @abc.abstractmethod
    def train_step(self, model: TModel, data: TData,
                   device: TDevice) -> dict[str, float]:
        """
        Perform one train step over the given data. Subclasses
        should implement this method to perform feed-forward
        and back-propagation. Moreover, the function should return
        a dictionary of metrics and their values to display the
        training progress.

        Parameters:
            model: nn.Module
                The model to be trained.
            data: Any
                A mini-batch returned by the train dataloader.
            device: torch.device or str
                Model's device.

        Returns: dict[str, float]
            Train metrics to be displayed in the progress bar.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def val_step(self, model: TModel, data: TData,
                 device: TDevice) -> dict[str, float]:
        """
        Perform one validation over the given data. Subclasses
        should implement this method to perform feed-forward
        over the data. The function should return a dictionary
        of metrics and their values to display the validation progress.
        The returned dictionary's keys will be prefixed with 'val_'
        before displaying, unless they already have that prefix.

        Parameters:
            model: nn.Module
                The model to be trained.
            data: Any
                A mini-batch returned by the validation dataloader.
            device: torch.device or str
                Model's device.

        Returns: dict[str, float]
            Validation metrics to be displayed in the progress bar.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def reset_train_metrics(self):
        """Reset training metrics, will be used at the start of an epoch."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def reset_val_metrics(self):
        """Reset validation metrics, will be used at the start of an epoch."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def compute_train_metrics(self) -> dict[str, float]:
        """
        Compute training metrics, will be used at the end of an epoch.

        Returns: dict[str, float]
            Training metrics to be displayed in the progress bar at the end of an epoch.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def compute_val_metrics(self) -> dict[str, float]:
        """
        Compute validation metrics, will be used at the end of an epoch.

        Returns: dict[str, float]
            Validation metrics to be displayed in the progress bar
            at the end of an epoch.
        """
        pass
