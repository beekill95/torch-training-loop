from __future__ import annotations

from typing import Optional
from typing import TypeVar
from typing import Union

import torch
from torch import nn

# Generics .
TModel = TypeVar('TModel', bound=nn.Module)
TData = TypeVar('TData')

# Possible device types.
# If it is an int, then it represents the ordinal of the cuda device.
# If it is None, it is cpu.
TDevice = Optional[Union[torch.device, str, int]]
