from src.agents.base import Agent
from src.services.openai_service import OpenAIService


class ClassifierAgent(Agent):
    def __init__(self, openai_service: OpenAIService):
        super().__init__("Classifier")
        self.openai = openai_service

    async def execute(self, query: str) -> str:
        """Classificar pergunta em: orientacao, resposta-cliente ou bug"""
        self.logger.info(f"[{self.name}] Classifying: {query[:50]}...")

        classification = await self.openai.classify(query)

        self.logger.info(f"[{self.name}] Result: {classification}")
        return classification
