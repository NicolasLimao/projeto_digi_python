from src.agents.base import Agent


class FormatterAgent(Agent):
    def __init__(self):
        super().__init__("Formatter")

    async def execute(self, response: str, mode: str = "orientacao") -> str:
        """
        Format response based on mode:
        - orientacao: bullet points/procedural
        - resposta-cliente: plain text for customer (WhatsApp)
        - bug: structured analysis with error details
        """
        self.logger.info(f"[{self.name}] Formatting response for mode: {mode}")

        if not response:
            return ""

        if mode == "orientacao":
            return self._format_orientacao(response)
        elif mode == "resposta-cliente":
            return self._format_resposta_cliente(response)
        elif mode == "bug":
            return self._format_bug(response)
        else:
            return response

    def _format_orientacao(self, response: str) -> str:
        """Format as procedural guidance with bullet points"""
        lines = response.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("-") or line.startswith("•"):
                formatted_lines.append(line)
            elif line.startswith(("1.", "2.", "3.", "a)", "b)", "c)")):
                formatted_lines.append(f"- {line}")
            else:
                formatted_lines.append(f"- {line}")

        return "\n".join(formatted_lines)

    def _format_resposta_cliente(self, response: str) -> str:
        """Format as customer-ready response (plain text for WhatsApp)"""
        lines = response.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("-") or line.startswith("•"):
                formatted_lines.append(line.lstrip("-•").strip())
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _format_bug(self, response: str) -> str:
        """Format as bug analysis with structure"""
        if "ERRO:" not in response and "ERROR:" not in response and "BUG:" not in response:
            response = f"BUG ENCONTRADO:\n{response}"

        sections = response.split("\n\n")
        formatted_sections = []

        for section in sections:
            if section.strip():
                formatted_sections.append(section.strip())

        return "\n\n".join(formatted_sections)
