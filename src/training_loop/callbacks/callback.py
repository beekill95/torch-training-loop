from __future__ import annotations

from typing import Generic, Literal, TYPE_CHECKING

from ..exceptions import StopTraining
from ..types import TModel

if TYPE_CHECKING:
    from ..training_loops.training_loop import TrainingLoop


class Callback(Generic[TModel]):

    def __init__(self):
        self._model = None
        self._loop = None

    def set_training_loop(self, loop: TrainingLoop):
        self._loop = loop

    @property
    def training_loop(self):
        return self._loop

    @property
    def model(self):
        return self._loop.model if self._loop is not None else None

    def on(self, event: Literal[
        'training_begin',
        'training_end',
        'epoch_begin',
        'epoch_end',
        'train_batch_begin',
        'train_batch_end',
        'val_batch_begin',
        'val_batch_end',
    ], **kwargs):
        handlers = {
            'training_begin': self.on_training_begin,
            'training_end': self.on_training_end,
            'epoch_begin': self.on_epoch_begin,
            'epoch_end': self.on_epoch_end,
            'train_batch_begin': self.on_train_batch_begin,
            'train_batch_end': self.on_train_batch_end,
            'val_batch_begin': self.on_val_batch_begin,
            'val_batch_end': self.on_val_batch_end,
        }

        if event not in handlers:
            raise ValueError(f'Unknown event: {event=}.')

        try:
            handlers[event](**kwargs)
        except StopTraining as e:
            if event == 'epoch_end':
                raise e

            print('WARN: StopTraining exception should only be raised'
                  ' in `on_epoch_end` callback.')

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, float]):
        pass

    def on_train_batch_begin(self, batch: int):
        pass

    def on_train_batch_end(self, batch: int, logs: dict[str, float]):
        pass

    def on_val_batch_begin(self, batch: int):
        pass

    def on_val_batch_end(self, batch: int, logs: dict[str, float]):
        pass
