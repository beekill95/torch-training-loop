from __future__ import annotations

import abc
from typing import Generic
from typing import TYPE_CHECKING

import torch

from ..types import TData
from ..types import TDevice

if TYPE_CHECKING:
    from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedTrainingStep(Generic[TData], abc.ABC):
    """
    An interface for distributed training logic on a single-node multi-gpu machine.
    """

    @abc.abstractmethod
    def init_distributed(self, model: DDP, device: TDevice):
        """Initialize the instance to be ready for training."""
        pass

    @abc.abstractmethod
    def train_step_distributed(self, model: DDP, data: TData,
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
            device: Model's device.

        Returns: dict[str, float]
            Train metrics to be displayed in the progress bar.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def val_step_distributed(self, model: DDP, data: TData,
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
            device: Model's device.

        Returns: dict[str, float]
            Validation metrics to be displayed in the progress bar.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def reset_train_metrics_distributed(self):
        """Reset training metrics of the current process at the start of an epoch."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def reset_val_metrics_distributed(self):
        """Reset validation metrics of the current process at the start of an epoch."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def compute_train_metrics_synced(self):
        """
        Sync & compute train metrics across processes at the end of an epoch.

        Returns: dict[str, float]
            Train metrics to be displayed in the progress bar at the end of epoch.
        """
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def compute_val_metrics_synced(self):
        """
        Sync & compute val metrics across processes at the end of an epoch.

        Returns: dict[str, float]
            Validation metrics to be displayed in the progress bar at the end of epoch.
        """
        pass
