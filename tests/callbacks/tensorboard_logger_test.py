from __future__ import annotations

from unittest.mock import Mock
from unittest.mock import patch

from torch.utils.tensorboard import SummaryWriter
from training_loop.callbacks import TensorBoardLogger


def test_log_metrics_at_every_batch():
    callback = TensorBoardLogger('./logs', update_freq=1, update_freq_unit='batch')

    with patch('torch.utils.tensorboard') as fake_tensorboard:
        fake_summary_writer = Mock(SummaryWriter)
        fake_tensorboard.SummaryWriter.return_value = fake_summary_writer

        callback.on('training_begin')

        for batch in range(10):
            f1 = batch * 0.1 + 0.02
            val_f1 = batch * 0.1 + 0.04

            callback.on('train_batch_end', batch=batch, logs={'f1': f1})
            fake_summary_writer.add_scalars.assert_called_once_with(
                'batch', {'f1': f1}, global_step=batch)
            fake_summary_writer.reset_mock()

            callback.on('val_batch_end', batch=batch, logs={'val_f1': val_f1})
            fake_summary_writer.add_scalars.assert_called_once_with(
                'batch', {'val_f1': val_f1}, global_step=batch)
            fake_summary_writer.reset_mock()

        # Epoch end will not log anything.
        callback.on('epoch_end', epoch=1, logs={'epoch': 1})
        fake_summary_writer.add_scalars.assert_not_called()


def test_log_metrics_at_fixed_batch_interval():
    callback = TensorBoardLogger('./logs', update_freq=5, update_freq_unit='batch')

    with patch('torch.utils.tensorboard') as fake_tensorboard:
        fake_summary_writer = Mock(SummaryWriter)
        fake_tensorboard.SummaryWriter.return_value = fake_summary_writer

        callback.on('training_begin')

        for batch in range(50):
            f1 = batch * 0.02 + 0.002
            val_f1 = batch * 0.1 + 0.004

            # Train batch.
            callback.on('train_batch_end', batch=batch, logs={'f1': f1})
            if batch % 5 == 0:
                fake_summary_writer.add_scalars.assert_called_once_with(
                    'batch', {'f1': f1}, global_step=batch)
                fake_summary_writer.reset_mock()
            else:
                fake_summary_writer.add_scalars.assert_not_called()

            # Val batch.
            callback.on('val_batch_end', batch=batch, logs={'val_f1': val_f1})
            if batch % 5 == 0:
                fake_summary_writer.add_scalars.assert_called_once_with(
                    'batch', {'val_f1': val_f1}, global_step=batch)
                fake_summary_writer.reset_mock()
            else:
                fake_summary_writer.add_scalars.assert_not_called()

        # Epoch end will not log anything.
        callback.on('epoch_end', epoch=1, logs={'epoch': 1})
        fake_summary_writer.add_scalars.assert_not_called()


def test_log_metrics_at_every_epoch():
    callback = TensorBoardLogger('./logs', update_freq=1, update_freq_unit='epoch')

    with patch('torch.utils.tensorboard') as fake_tensorboard:
        fake_summary_writer = Mock(SummaryWriter)
        fake_tensorboard.SummaryWriter.return_value = fake_summary_writer

        callback.on('training_begin')

        for epoch in range(10):
            for batch in range(10):
                callback.on('train_batch_end', batch=batch, logs={'batch': batch})
                fake_summary_writer.add_scalars.assert_not_called()

                callback.on('val_batch_end', batch=batch, logs={'batch': batch})
                fake_summary_writer.add_scalars.assert_not_called()

            # Epoch end will not log anything.
            callback.on('epoch_end', epoch=1, logs={'epoch': 1})
            fake_summary_writer.add_scalars.assert_called_once_with(
                'epoch', {'epoch': 1}, global_step=epoch)
            fake_summary_writer.reset_mock()


def test_log_metrics_at_fixed_epoch_interval():
    callback = TensorBoardLogger('./logs', update_freq=5, update_freq_unit='epoch')

    with patch('torch.utils.tensorboard') as fake_tensorboard:
        fake_summary_writer = Mock(SummaryWriter)
        fake_tensorboard.SummaryWriter.return_value = fake_summary_writer

        callback.on('training_begin')

        for epoch in range(50):
            for batch in range(10):
                # Batch ends will not log anything.
                callback.on('train_batch_end', batch=batch, logs={'batch': batch})
                fake_summary_writer.add_scalars.assert_not_called()

                callback.on('val_batch_end', batch=batch, logs={'batch': batch})
                fake_summary_writer.add_scalars.assert_not_called()

            callback.on('epoch_end', epoch=1, logs={'epoch': 1})
            if epoch % 5 == 0:
                fake_summary_writer.add_scalars.assert_called_once_with(
                    'epoch', {'epoch': 1}, global_step=epoch)
                fake_summary_writer.reset_mock()
            else:
                fake_summary_writer.add_scalars.assert_not_called()


def test_close_writer():
    callback = TensorBoardLogger('./logs', update_freq=5, update_freq_unit='batch')

    with patch('torch.utils.tensorboard') as fake_tensorboard:
        fake_summary_writer = Mock(SummaryWriter)
        fake_tensorboard.SummaryWriter.return_value = fake_summary_writer

        # Init callback.
        callback.on('training_begin')

        # Stop training.
        callback.on('training_end')
        fake_summary_writer.close.assert_called_once()
