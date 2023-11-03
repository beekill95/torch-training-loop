from __future__ import annotations

from torch import nn
from typing import TypeVar

# Generics .
TModel = TypeVar('TModel', bound=nn.Module)
TData = TypeVar('TData')
