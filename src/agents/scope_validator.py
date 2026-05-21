from typing import Dict, Any
from src.agents.base import Agent
from src.services.openai_service import OpenAIService


class ScopeValidatorAgent(Agent):
    def __init__(self, openai_service: OpenAIService):
        super().__init__("ScopeValidator")
        self.openai = openai_service

    async def execute(self, query: str) -> Dict[str, Any]:
        """Validar se pergunta está dentro do escopo Digisac"""
        self.logger.info(f"[{self.name}] Validating scope: {query[:50]}...")

        result = await self.openai.validate_scope(query)

        self.logger.info(f"[{self.name}] Result: {result}")
        return result
