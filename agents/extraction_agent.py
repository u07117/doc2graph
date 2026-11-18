import json

from agents.base_agent import BaseAgent
from strategies import get_strategy


class ExtractionAgent(BaseAgent):
    """
    Extracts a graph fragment from one document chunk.

    The strategy provides the explicit system prompt for the selected graph type.
    """

    def run(self, document_type, plan, chunk, chunk_number):
        strategy = get_strategy(document_type)
        system_prompt = strategy.extraction_prompt()

        user_prompt = f"""
DOCUMENT TYPE:
{document_type}

EXTRACTION PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

CHUNK NUMBER:
{chunk_number}

TASK:
Extract only nodes and relationships supported by this excerpt.
Preserve page and section markers when present. Include concise evidence.
Do not assume two items are connected merely because they appear next to
each other in the document.

DOCUMENT EXCERPT:
{chunk}
"""
        result = self.ask_json(system_prompt, user_prompt)
        result["source_chunk"] = chunk_number
        return result
