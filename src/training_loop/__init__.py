from importlib.metadata import version, PackageNotFoundError

from .training_loops import (
    SimpleTrainingLoop,
    TrainingLoop,
    TrainingStep,
    SimpleTrainingStep,
)

try:
    __version__ = version('torch_training_loop')
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
