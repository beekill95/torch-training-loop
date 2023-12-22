from __future__ import annotations

from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from training_loop.progress_reporter import format_metrics
from training_loop.progress_reporter import ProgressReporter


@pytest.mark.parametrize("verbose", [0, 1, 2, 5, 10, 30, 50])
@patch("training_loop.progress_reporter.tqdm")
def test_init(tqdm, verbose):
    ProgressReporter(epoch=1, total_batches=100, total_epochs=10, verbose=verbose)

    if verbose == 1:
        tqdm.assert_called_once_with(total=100, desc="Epoch 1/10")
    else:
        tqdm.assert_not_called()


@pytest.mark.parametrize("verbose", [0, 1, 2, 5, 10, 30, 50])
@patch("training_loop.progress_reporter.tqdm")
def test_next_batch(tqdm, verbose):
    bar = MagicMock()
    tqdm.return_value = bar

    reporter = ProgressReporter(
        epoch=1,
        total_batches=100,
        total_epochs=10,
        verbose=verbose,
    )
    reporter.next_batch()

    if verbose == 1:
        bar.update.assert_called_once()
    else:
        bar.update.assert_not_called()


@pytest.mark.parametrize("verbose", [0, 1, 2, 5, 10, 30, 50])
@patch("training_loop.progress_reporter.tqdm")
def test_close_report(tqdm, verbose):
    bar = MagicMock()
    tqdm.return_value = bar

    reporter = ProgressReporter(
        epoch=1,
        total_batches=100,
        total_epochs=10,
        verbose=verbose,
    )
    reporter.close_report()

    if verbose == 1:
        bar.close.assert_called_once()
    else:
        bar.close.assert_not_called()


@pytest.mark.parametrize("verbose", [0, 1, 2, 5, 10, 30, 50])
@patch("training_loop.progress_reporter.tqdm")
def test_report_batch_progress(tqdm, verbose):
    bar = MagicMock()
    tqdm.return_value = bar

    reporter = ProgressReporter(
        epoch=1,
        total_batches=100,
        total_epochs=10,
        verbose=verbose,
    )

    reporter.report_batch_progress("Training", {"f1": 0.1, "loss": 0.9})
    reporter.report_batch_progress("Training", {"f1": 0.2, "loss": 0.8})
    reporter.report_batch_progress("Training", {"f1": 0.3, "loss": 0.7})
    reporter.report_batch_progress("Validation", {"val_f1": 0.4, "val_loss": 0.6})
    reporter.report_batch_progress("Validation", {"val_f1": 0.55, "val_loss": 0.45})

    if verbose == 1:
        bar.set_description.assert_has_calls(
            [
                call("Epoch 1/10 - Training"),
                call("Epoch 1/10 - Validation"),
            ],
            any_order=False,
        )
        bar.set_postfix_str.assert_has_calls(
            [
                call("f1=0.1000; loss=0.9000"),
                call("f1=0.2000; loss=0.8000"),
                call("f1=0.3000; loss=0.7000"),
                call("val_f1=0.4000; val_loss=0.6000"),
                call("val_f1=0.5500; val_loss=0.4500"),
            ],
            any_order=False,
        )
    else:
        bar.assert_not_called()


@pytest.mark.parametrize("epoch", [1, 5, 10, 50, 100])
@pytest.mark.parametrize("verbose", [0, 1, 2, 5, 10])
@patch("training_loop.progress_reporter.tqdm")
def test_report_epoch_progress(tqdm, capsys, epoch, verbose):
    bar = MagicMock()
    tqdm.return_value = bar

    reporter = ProgressReporter(
        epoch=epoch,
        total_batches=100,
        total_epochs=100,
        verbose=verbose,
    )

    reporter.report_epoch_progress(
        "Finished",
        {
            "f1": 0.9,
            "loss": 0.1,
            "val_f1": 0.89,
            "val_loss": 0.11,
        },
    )

    stdout_str = capsys.readouterr().out

    if verbose == 0:
        bar.set_description.assert_not_called()
        bar.set_postfix_str.assert_not_called()
        assert stdout_str == ""
    elif verbose == 1:
        bar.set_description.assert_called_once_with(f"Epoch {epoch}/100 - Finished")
        bar.set_postfix_str.assert_called_once_with(
            "f1=0.9000; loss=0.1000; val_f1=0.8900; val_loss=0.1100")
    elif verbose == 2:
        assert stdout_str == (
            f"Epoch {epoch}/100 - Finished: "
            "f1=0.9000; loss=0.1000; val_f1=0.8900; val_loss=0.1100\n")
    elif verbose > 2:
        if epoch == 1 or epoch == 100 or (epoch % verbose == 0):
            assert stdout_str == (
                f"Epoch {epoch}/100 - Finished: "
                "f1=0.9000; loss=0.1000; val_f1=0.8900; val_loss=0.1100\n")
        else:
            assert stdout_str == ""


def test_format_metrics_large_floats():
    metrics = {
        "v1": 1234.5678,
        "v2": 12345.678,
        "v3": 123456.78,
        "v4": 1234567.8,
    }
    result = format_metrics(metrics)
    assert result == "v1=1.2346e+03; v2=1.2346e+04; v3=1.2346e+05; v4=1.2346e+06"


def test_format_metrics_small_floats():
    metrics = {
        "v1": 0.00012345,
        "v2": 0.000012345,
        "v3": 0.0000012345,
        "v4": 0.00000012345,
    }
    result = format_metrics(metrics)
    assert result == "v1=1.2345e-04; v2=1.2345e-05; v3=1.2345e-06; v4=1.2345e-07"


def test_format_metrics_normal_floats():
    metrics = {
        "v1": 123.45678,
        "v2": 12.345678,
        "v3": 1.2345678,
        "v4": 0.1234567,
        "v5": 0.0123456,
        "v6": 0.0012345,
    }
    result = format_metrics(metrics)
    assert (
        result == "v1=123.4568; v2=12.3457; v3=1.2346; v4=0.1235; v5=0.0123; v6=0.0012")
