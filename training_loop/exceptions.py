class StopTraining(Exception):
    """Stop the training loop, should only be used in `on_epoch_end` callback."""
    pass
