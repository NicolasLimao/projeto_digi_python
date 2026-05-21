from abc import ABC, abstractmethod
from typing import Any, Dict
from src.logger import get_logger

logger = get_logger(__name__)


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logger

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass
