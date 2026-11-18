from agents.base_agent import BaseAgent


class ClassifierAgent(BaseAgent):
    """Classifies documents into supported graph categories."""

    SYSTEM_PROMPT = """
You are a Document Classification Agent.

Classify the supplied document into exactly ONE of the following categories.

1. business_process
   Use for SOPs, policies, procedures, operational workflows, approval
   processes, claims handling, onboarding, payment operations, and documents
   describing activities, actors, decisions, conditions, hand-offs or outcomes.

2. requirements
   Use for product requirement documents, functional/non-functional
   requirements, user stories, feature descriptions, constraints, acceptance
   criteria and explicit feature dependencies.

3. generic_knowledge
   Use for reports, research papers, explanatory documents, reference
   material, and content that does not clearly fit business_process or
   requirements.

IMPORTANT:
- Do not infer a workflow merely because the document has numbered sections.
- For mixed or unclear documents, choose generic_knowledge and explain why.
- Base classification on document content, not just its filename.

Return exactly this JSON structure:
{
  "document_type": "business_process | requirements | generic_knowledge",
  "confidence": 0.0,
  "reason": "Brief explanation based on document content",
  "recommended_graph": "What nodes and relationships should represent"
}

Confidence must be a number between 0 and 1. Do not invent information.
"""

    def run(self, document_name, preview):
        """preview: excerpts sampled across the whole document
        (core.chunker.build_preview), not just its beginning."""
        user_prompt = f"""
DOCUMENT NAME:
{document_name}

DOCUMENT CONTENT PREVIEW (excerpts sampled across the whole document):
{preview}
"""
        result = self.ask_json(self.SYSTEM_PROMPT, user_prompt)

        supported_types = {
            "business_process",
            "requirements",
            "generic_knowledge",
        }

        if result.get("document_type") not in supported_types:
            result["document_type"] = "generic_knowledge"
            result["reason"] = (
                result.get("reason", "")
                + " Unsupported category; defaulted to generic_knowledge."
            ).strip()

        return result
