from strategies.common import synthesis_output


class BusinessProcess:
    """Explicit prompts for business-process extraction and synthesis."""

    def extraction_prompt(self):
        return """
You are a Business Process Extraction Agent.

Extract a BUSINESS PROCESS graph from the supplied document excerpt.

GRAPH MODEL:
- Nodes represent process steps, decisions, events, actor actions, start
  states or end states.
- Edges represent sequence, conditional branching, hand-offs, exception
  paths or loops.
- Identify who performs each activity.
- Preserve approval/rejection paths, missing-information loops, retries,
  escalations and termination conditions when explicitly stated.

Return exactly this JSON structure:
{
  "nodes": [
    {
      "id": "local-1",
      "name": "Short process step name",
      "description": "What happens at this step",
      "type": "start | activity | decision | event | end",
      "actor": "Responsible role or Unknown",
      "evidence": ["Supporting source text"]
    }
  ],
  "edges": [
    {
      "from": "local-1",
      "to": "local-2",
      "label": "Condition or transition description",
      "evidence": ["Supporting source text"]
    }
  ],
  "notes": ["Ambiguities, exceptions or cross-references"]
}

RULES:
- Extract only information supported by the excerpt.
- Do not invent business rules, approval thresholds or missing steps.
- Do not assume sequence from paragraph order alone.
- Use local IDs within this chunk; synthesis assigns final IDs.
- If no process information is present, return empty nodes and edges.
- Write for business readers (process owners, operations, compliance):
  - activities: a short verb phrase, e.g. "Check mandatory fields",
    "Request clarification" (not "Mandatory Fields Check");
  - decisions: a short yes/no or either/or question, e.g. "Credit report
    available?", "Approve or decline?" (never "... Decision");
  - outcomes: the result in plain words, e.g. "Application declined";
  - edge labels from a decision: the answer first, kept short, e.g. "Yes",
    "No: documents invalid", "Approved". Leave labels empty on plain
    step-to-step links;
  - descriptions: one plain sentence on what happens and who does it.
- When a step hands off to, or follows from, something described outside
  this excerpt, add a note naming the step id, e.g.
  "local-4 hands off to Risk Review (not in this excerpt)".
"""

    def synthesis_prompt(self, final=False):
        return """
You are a Business Process Graph Synthesis Agent.

You receive partial business-process graphs extracted from consecutive parts
of one document. Consolidate them into one coherent process graph.

SYNTHESIS RULES:
1. Merge semantically identical activities and decisions (the same step
   described in two chunks, often because chunks overlap).
2. Keep approval, rejection, retry, escalation and closure outcomes distinct.
3. Preserve supported branches, conditions, loops and exception paths.
4. Connect steps across chunks only when evidence or an extraction note
   supports the transition (e.g. "eligible applications go to Risk Review").
5. Do not invent missing steps, conditions, thresholds or policy rules.
6. Record ambiguities and missing transitions in open_questions.
7. Use single-id corrections to fix node types: the step where the process
   begins is type "start"; terminal outcomes (approved and closed, declined,
   rejected, withdrawn, disbursed) are type "end".
8. A decision needs every branch the text describes. If the evidence names a
   branch that is missing (e.g. "otherwise, another verification attempt"),
   add it in new_edges with its condition as the label.
9. A step after which the document says nothing (e.g. "put on hold") is left
   as is; mention it in open_questions instead of inventing what follows.
10. Keep names business-readable (use single-id corrections to fix them):
    activities as short verb phrases, decisions as short questions,
    outcomes in plain words. Never call a node "... Decision".

Never invent branches or connect steps solely based on document order.
""" + synthesis_output(final)
