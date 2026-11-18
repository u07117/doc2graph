"""Output format shared by every strategy's synthesis prompt.

Synthesis returns DECISIONS (which nodes to merge, which edges to add), not a
rewritten graph. The code in core/graph_utils.py applies them and keeps every
node that is not merged, so nothing is silently lost and the reply stays small
however large the document is.
"""

SYNTHESIS_OUTPUT = """
OUTPUT
Do NOT rewrite the graph. Return only your decisions; the system applies them
and keeps every node you do not merge.

Return exactly this JSON structure:
{
  "merges": [
    {
      "ids": ["C2:local-1", "C5:local-3"],
      "name": "Best combined name",
      "description": "Combined description",
      "type": "Optional: corrected node type",
      "actor": "Optional: corrected actor"
    }
  ],
  "new_edges": [
    {
      "from": "C2:local-4",
      "to": "C3:local-1",
      "label": "Condition or relationship",
      "evidence": ["Supporting source text"]
    }
  ],
  "open_questions": ["Ambiguities requiring human review"]
}

- merges: groups of 2 or more ids that refer to the same thing. Copy ids
  exactly as listed. An id may appear in at most one group.
  A group with ONE id corrects that node without merging it, e.g.
  {"ids": ["C1:local-1"], "type": "start"} or a clearer "name".
- new_edges: relationships that are missing, mainly between nodes from
  different chunks (different C numbers), supported by evidence or by the
  extraction notes. Copy ids exactly. Do not repeat existing edges.
- Use empty lists when nothing applies.
- open_questions (and assumptions) are read by business users: write plain
  sentences that refer to steps by name. Never mention ids such as
  "C2:local-1", chunks, excerpts, nodes or edges.
"""

FINAL_FIELDS = """
This is the FINAL pass over the complete graph. Also include these keys:
  "graph_title": "Title for the graph",
  "description": "Short overview",
  "assumptions": ["Explicitly identified inferences"]
"""


def synthesis_output(final=False):
    return SYNTHESIS_OUTPUT + (FINAL_FIELDS if final else "")
