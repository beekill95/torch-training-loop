from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .training_loops import SimpleTrainingStep
from .training_loops import TrainingLoop
from .training_loops import TrainingStep

try:
    __version__ = version('torch_training_loop')
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
