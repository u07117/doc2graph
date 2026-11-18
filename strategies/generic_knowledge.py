from strategies.common import synthesis_output


class GenericKnowledge:
    """Explicit prompts for general entity and relationship graphs."""

    def extraction_prompt(self):
        return """
You are a Generic Knowledge Graph Extraction Agent.

Extract a concise KNOWLEDGE GRAPH from the supplied document excerpt.

GRAPH MODEL:
- Nodes represent important entities, concepts, people or roles,
  organizations, events, facts or claims.
- Edges represent explicitly stated relationships between those nodes.
- Prefer meaningful concepts over extracting every noun.
- Do not force content into a business workflow.

Return exactly this JSON structure:
{
  "nodes": [
    {
      "id": "local-1",
      "name": "Entity or concept",
      "description": "Supported description",
      "type": "entity | concept | person_or_role | organization | event | fact | other",
      "actor": "Relevant owner or Unknown",
      "evidence": ["Supporting source text"]
    }
  ],
  "edges": [
    {
      "from": "local-1",
      "to": "local-2",
      "label": "Explicit relationship",
      "evidence": ["Supporting source text"]
    }
  ],
  "notes": ["Ambiguities or context"]
}

RULES:
- Extract only important information and explicitly supported relationships.
- Do not invent causal, ownership or dependency relationships.
- If the excerpt has little relational content, return a small graph or empty
  nodes and edges rather than fabricating structure.
"""

    def synthesis_prompt(self, final=False):
        return """
You are a Generic Knowledge Graph Synthesis Agent.

You receive partial knowledge graphs extracted from consecutive parts of one
document. Consolidate them.

SYNTHESIS RULES:
1. Merge entities and concepts referring to the same thing; normalize names
   without changing their meaning.
2. Add cross-chunk relationships only when supported by source evidence.
3. Do not force the document into a process flow.
4. Do not invent relationships or unsupported facts.
""" + synthesis_output(final)
