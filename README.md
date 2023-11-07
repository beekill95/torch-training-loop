[![Tests](https://github.com/beekill95/torch-training-loop/workflows/Tests/badge.svg)](https://github.com/beekill95/torch-training-loop/actions?query=workflow:"Tests")
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-training-loop)](https://pypi.org/project/torch-training-loop/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-training-loop)

⚠️The package is under development, expect bugs and breaking changes!

# Torch Training Loop

Simple Keras-inspired Training Loop for Pytorch.

## Installation

> pip install torch-training-loop

## Features

* Simple API for training Torch models;
* Support training `DataParallel` and `DistributedDataParallel` models;
* Support Keras-like callbacks for logging metrics to Tensorboard, model checkpoint,
and early stopping;
* Show training & validation progress via `tqdm`;
* Display metrics during training & validation via `torcheval`.

## Usage

This package consists of two main classes for training Torch models:
`TrainingLoop` and `SimpleTrainingStep`.
In order to train a torch model, you need to initiate these two classes:

```python
import torch
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy
from training_loop import TrainingLoop, SimpleTrainingStep
from training_loop.callbacks import EarlyStopping

model = ...
# Support training DataParallel models.
# model = DataParallel(model)

train_dataloader = ...
val_dataloader = ...

loop = TrainingLoop(
    model,
    step=SimpleTrainingStep(
        optimizer_fn=lambda params: Adam(params, lr=0.0001),
        loss=torch.nn.CrossEntropyLoss(),
        metrics=('accuracy', MulticlassAccuracy(num_classes=10)),
    ),
    device='cuda',
)
loop.fit(
    train_dataloader,
    val_dataloader,
    epochs=10,
    callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', patience=20),
    ],
)
```

In the above example, initializing the `SimpleTrainingStep` class and
calling the `fit()` method of the `TrainingLoop` class are very similar to that of Keras API.
Additionally, you can also train `DistributedDataParallel` models to utilize multigpus setup.
Currently, it only supports training on single-node multigpus machines.

```python
from contextlib import contextmanager
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy
from training_loop import SimpleTrainingStep
from training_loop.distributed import DistributedTrainingLoop


@contextmanager
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    try:
        yield
    finally:
        os.environ.pop('MASTER_ADDR')
        os.environ.pop('MASTER_PORT')
        dist.destroy_process_group()


def train_ddp(rank, world_size):
    with setup_ddp(rank, world_size):
        model = ...
        model = DDP(model, device_ids=[rank])

        train_loader = ...
        val_loader = ...

        loop = DistributedTrainingLoop(
            model,
            step=SimpleTrainingStep(
                optimizer_fn=lambda params: Adam(params, lr=0.0001),
                loss=torch.nn.CrossEntropyLoss(),
                metrics=('accuracy', MulticlassAccuracy(num_classes=10)),
            ),
            device=rank,
            rank=rank,
        )

        loop.fit(train_loader, val_loader, epochs=1)


def main():
    world_size = torch.cuda.device_count()

    mp.spawn(
        train_ddp,
        args=(world_size, ),
        nprocs=world_size,
        join=True,
    )

    return 0


if __name__ == '__main__':
    exit(main())
```

You can find more examples and documentation in the source code and in the `examples` folder.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
