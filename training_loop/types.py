from __future__ import annotations

import torch
from torch import nn
from torcheval.metrics import Metric
from typing import TypeVar, Tuple

# Generics .
TModel = TypeVar('TModel', bound=nn.Module)

# Metrics.
NamedMetric = Tuple[str, Metric[torch.Tensor]]
