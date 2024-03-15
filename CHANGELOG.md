# Changelog

## Upcoming

* Implement `StateSerializable` interface for training loops and training steps to support training loop checkpoint.
* Implement evaluation loop.
* Support mixed-precision training.
* Improve logging.

## [v0.1.3](https://github.com/beekill95/torch-training-loop/releases/tag/v0.1.3) - 2023-12-22

### Fixed

* Not passing metrics to SimpleTrainingStep won't cause initialization error.
([#26](https://github.com/beekill95/torch-training-loop/issues/26))
* Displaying scientific notation when metrics values becoming too large or too small.
([#29](https://github.com/beekill95/torch-training-loop/issues/29))

## [v0.1.2](https://github.com/beekill95/torch-training-loop/releases/tag/v0.1.2) - 2023-11-27

### Changed

* Support dataloaders without `__len__` implementation. ([#21](https://github.com/beekill95/torch-training-loop/pull/21))

### Added

* Different verbose levels for training loops' `fit()` function. ([#20](https://github.com/beekill95/torch-training-loop/pull/20))

## [v0.1.1](https://github.com/beekill95/torch-training-loop/releases/tag/v0.1.1) - 2023-11-11

### Added

* Implement distributed training loop. ([#12](https://github.com/beekill95/torch-training-loop/pull/12))

## [v0.1.0](https://github.com/beekill95/torch-training-loop/releases/tag/v0.1.0) - 2023-11-03

`TrainingLoop` instances can be initialized directly.
They receives an instance of `TrainingStep` specifying the logic of training/validating a model.

### Removed

* __Breaking__: Remove `SimpleTrainingLoop`, replace it with `SimpleTrainingStep`.
([#11](https://github.com/beekill95/torch-training-loop/pull/11))
* Remove `tensorboard` dependency.

### Added

* Add usage section in README.
* Add an example for training conditional GAN on Fashion MNIST with custom loop.

## [v0.0.3](https://github.com/beekill95/torch-training-loop/releases/tag/v0.0.3) - 2023-10-25

### Added

* References to both training loop and model instances in callbacks.

## [v0.0.2](https://github.com/beekill95/torch-training-loop/releases/tag/v0.0.2) - 2023-10-24

### Added

* Update progress bar after each epoch finishes.

### Fixed

* Cuda training not working.

## [v0.0.1](https://github.com/beekill95/torch-training-loop/releases/tag/v0.0.1) - 2023-10-24

_Initial Release._
