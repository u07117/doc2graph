import json
from concurrent.futures import ThreadPoolExecutor

from agents.base_agent import BaseAgent
from core.graph_utils import apply_synthesis, combine, compact_listing
from strategies import get_strategy


class SynthesisAgent(BaseAgent):
    """Merges chunk-level graph fragments using hierarchical reduction.

    At each level, consecutive fragments are combined in batches. The model
    sees a compact listing of the combined graph and returns only merge
    decisions and missing edges; code applies them (core/graph_utils.py).
    So no node is lost, and replies stay small even for very large documents.
    """

    def __init__(self, client, model, batch_size=8, concurrency=4):
        super().__init__(client, model)
        self.batch_size = max(2, batch_size)
        self.concurrency = max(1, concurrency)

    def merge(self, document_type, plan, graph, final=False):
        if not graph["nodes"]:
            graph["open_questions"].append("No graph fragments were extracted.")
            return graph

        strategy = get_strategy(document_type)
        system_prompt = strategy.synthesis_prompt(final=final)
        chunks = sorted({c for n in graph["nodes"] for c in n["source_chunks"]})

        user_prompt = f"""
DOCUMENT TYPE:
{document_type}

EXTRACTION PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

GRAPH TO CONSOLIDATE ({len(graph['nodes'])} nodes from chunks {chunks[0]}-{chunks[-1]}):
Node ids are "C<chunk number>:<local id>".

{compact_listing(graph)}

TASK:
- Identify nodes that describe the same thing and should be merged.
- Identify missing relationships, mainly across chunks, that the evidence or
  notes support.
- Put ambiguities and unresolved relationships in open_questions.
"""
        result = self.ask_json(system_prompt, user_prompt)
        merged = apply_synthesis(graph, result)

        if final:
            for key in ("graph_title", "description"):
                if result.get(key):
                    merged[key] = str(result[key])
            merged["assumptions"] = [str(a) for a in result.get("assumptions") or [] if a]
        return merged

    def reduce_level(self, document_type, plan, graphs):
        """One level of the reduction. Returns (graphs, done)."""
        if len(graphs) <= self.batch_size:
            return [self.merge(document_type, plan, combine(graphs), final=True)], True

        batches = [
            combine(graphs[start:start + self.batch_size])
            for start in range(0, len(graphs), self.batch_size)
        ]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(batches))) as pool:
            merged = list(pool.map(lambda g: self.merge(document_type, plan, g), batches))
        return merged, False

    def run(self, document_type, plan, graphs):
        """Reduce normalized fragment graphs (in document order) to one graph."""
        graphs = graphs or [combine([])]
        done = False
        while not done:
            graphs, done = self.reduce_level(document_type, plan, graphs)
        return graphs[0]
