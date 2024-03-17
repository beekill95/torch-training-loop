from __future__ import annotations

import abc
from typing import Generic
from typing import Self

from ..types import TStateDict


class StateSerializable(Generic[TStateDict], abc.ABC):
    """
    Interface for serialize/deserialize the internal state of training loops & steps.
    """

    @abc.abstractmethod()
    def state_dict(self) -> TStateDict:
        pass

    @abc.abstractmethod()
    def load_state_dict(self, state_dict: TStateDict) -> Self:
        pass
