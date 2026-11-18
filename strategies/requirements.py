from strategies.common import synthesis_output


class Requirements:
    """Explicit prompts for requirements and dependency graph extraction."""

    def extraction_prompt(self):
        return """
You are a Requirements Extraction Agent.

Extract a REQUIREMENTS AND DEPENDENCY graph from the supplied excerpt.

GRAPH MODEL:
- Nodes represent requirements, features, user stories, actors, constraints,
  acceptance criteria or external dependencies.
- Edges represent depends-on, implements, constrains, owned-by or
  acceptance-for relationships.

Return exactly this JSON structure:
{
  "nodes": [
    {
      "id": "local-1",
      "name": "Requirement or feature name",
      "description": "Requirement details",
      "type": "requirement | feature | actor | constraint | acceptance_criterion | dependency",
      "actor": "Owner or Unknown",
      "evidence": ["Supporting source text"]
    }
  ],
  "edges": [
    {
      "from": "local-1",
      "to": "local-2",
      "label": "Relationship type and direction",
      "evidence": ["Supporting source text"]
    }
  ],
  "notes": ["Ambiguities or missing details"]
}

RULES:
- Extract explicit requirements and dependencies only.
- Preserve requirement IDs when the document provides them.
- Do not invent acceptance criteria, owners or dependencies.
- If no requirements are present, return empty nodes and edges.
"""

    def synthesis_prompt(self, final=False):
        return """
You are a Requirements Graph Synthesis Agent.

You receive partial requirement and dependency graphs extracted from
consecutive parts of one document. Consolidate them.

SYNTHESIS RULES:
- Merge semantically identical requirements and features.
- Preserve source requirement IDs where meaningful (keep them in the name).
- Preserve the direction and meaning of dependencies.
- Keep constraints and acceptance criteria connected to relevant features.
- Add cross-chunk relationships only when evidence supports them.
- Do not invent requirements or relationships.
""" + synthesis_output(final)
