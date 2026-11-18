import json

from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Creates extraction guidance for the classified document."""

    SYSTEM_PROMPT = """
You are a Document Analysis and Planning Agent.

You receive the document classification and a preview of the original document.
Prepare a focused extraction plan for the selected graph type.

Instructions:
1. Identify the document's purpose and scope.
2. Identify relevant actors, roles, entities or stakeholders.
3. Identify important terminology and definitions.
4. Explain what nodes should represent for this document type.
5. Explain what relationships or edges should represent.
6. Preserve explicitly stated conditions, exceptions, dependencies and loops.
7. Do not invent facts, steps, rules, thresholds or relationships.

Return exactly this JSON structure:
{
  "document_title": "Document title",
  "objective": "What the document is about",
  "actors_or_entities": ["Actor or entity"],
  "key_terms": ["Important term"],
  "extraction_guidance": "Specific instructions for extracting this document"
}
"""

    def run(self, classification, preview):
        """preview: excerpts sampled across the whole document."""
        user_prompt = f"""
DOCUMENT CLASSIFICATION:
{json.dumps(classification, ensure_ascii=False, indent=2)}

DOCUMENT CONTENT PREVIEW (excerpts sampled across the whole document):
{preview}
"""
        return self.ask_json(self.SYSTEM_PROMPT, user_prompt)
