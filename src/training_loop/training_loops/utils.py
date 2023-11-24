from __future__ import annotations

TRAIN_DATALOADER_SEPARATOR = ()


def train_dataloader_separator():
    yield (-1, TRAIN_DATALOADER_SEPARATOR)


def prefix_val_metrics_keys(metrics: dict[str, float], prefix: str) -> dict[str, float]:

    def prefix_key(key):
        return (key if key.startswith(prefix) else f'{prefix}{key}')

    return {prefix_key(k): v for k, v in metrics.items()}


def hasfunc(obj: object, funcname: str):
    func = getattr(obj, funcname, None)
    return callable(func)
