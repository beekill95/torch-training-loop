from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from types import TracebackType


class ProgressReporter(AbstractContextManager):
    """
    A thin wrapper around tqdm progress bar implements logic for reporting
    training and validation progress with different verbose levels.
    """

    def __init__(
            self,
            epoch: int,
            *,
            total_epochs: int,
            total_batches: int | float = float('inf'),
            verbose: int = 1,
    ) -> None:
        """
        Construct progress reporter instance for an epoch.

        Parameters:
            epoch: the current epoch.
            total_epochs: the number of epochs.
            total_batches: the number of batches within the epoch.
            verbose: verbose level.
                If verbose < 1: no progress bar is displayed.
                If verbose = 1: progress bar is displayed and the batches/epoch
                    train and validation metrics are reported.
                If verbose = 2: no progress bar is displayed, and only train/validation
                    for the epoch is reported.
                If verbose > 2: train/validation metrics for the epoch is reported
                    only if `epoch` is multiple times of `verbose`.
        """
        self._epoch = epoch
        self._epochs = total_epochs
        self._verbose = verbose

        if verbose == 1:
            self._bar = tqdm(total=total_batches, desc=self._get_epoch_description())
        else:
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,  # noqa
            exc_value: BaseException | None,  # noqa
            traceback: TracebackType | None,  # noqa
    ) -> bool | None:
        self.close_report()

    def next_batch(self):
        if self._bar:
            self._bar.update()

    def report_epoch_progress(self, desc: str, metrics: dict[str, float]) -> None:
        verbose = self._verbose
        epoch = self._epoch
        epochs = self._epochs

        if verbose < 1:
            return

        if self._bar:
            self._set_bar_desciption(desc)
            self._bar.set_postfix_str(format_metrics(metrics))
        elif epoch in [1, epochs] or verbose == 2 or (epoch % verbose == 0):
            print(
                self._get_epoch_description(desc),
                ': ',
                format_metrics(metrics),
                sep='',
            )

    def report_batch_progress(self, desc: str, metrics: dict[str, float]) -> None:
        if self._bar:
            self._set_bar_desciption(desc)
            self._bar.set_postfix_str(format_metrics(metrics))

    def close_report(self):
        if self._bar:
            self._bar.close()

    def _get_epoch_description(self, desc: str | None = None):
        epoch_str = f'Epoch {self._epoch}/{self._epochs}'
        return f'{epoch_str} - {desc}' if desc else epoch_str

    def _set_bar_desciption(self, desc):
        if not hasattr(self, '_cur_desc'):
            self._cur_desc = None

        if self._bar and self._cur_desc != desc:
            self._bar.set_description(self._get_epoch_description(desc))
            self._cur_desc = desc


def format_metrics(metrics: dict[str, float]) -> str:
    strs = (f'{k}={v:.4f}' for k, v in metrics.items())
    return '; '.join(strs)
