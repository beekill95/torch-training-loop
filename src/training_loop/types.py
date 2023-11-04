from __future__ import annotations

import torch
from torch import nn
from typing import Optional, TypeVar, Union

# Generics .
TModel = TypeVar('TModel', bound=nn.Module)
TData = TypeVar('TData')

# Possible device types.
# If it is an int, then it represents the ordinal of the cuda device.
# If it is None, it is cpu.
TDevice = Optional[Union[torch.device, str, int]]
