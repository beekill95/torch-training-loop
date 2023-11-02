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
* Support training `DataParallel` models;
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
        optimizer_fn=lambda params: Adam(params),
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
You can find more examples and documentation in the source code and in the `examples` folder.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
