from importlib.metadata import version, PackageNotFoundError

from .training_loops import SimpleTrainingLoop, TrainingLoop

try:
    __version__ = version('training_loop')
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
