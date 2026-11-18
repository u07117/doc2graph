"""OpenAI-compatible fake client (chat.completions.create) for offline tests.

Extraction: each sentence of the chunk becomes a node, linked in order.
Synthesis: merges nodes with identical names and links the last node of each
chunk to the first node of the next one. Tests plumbing, not model quality.
"""
import json
import re
import threading
from types import SimpleNamespace as NS


class FakeClient:
    def __init__(self, fail_on_chunks=()):
        self.chat = NS(completions=NS(create=self.create))
        self.fail_on_chunks = set(fail_on_chunks)
        self.calls = {"classify": 0, "plan": 0, "extract": 0, "synthesize": 0}
        self.extracted, self.reply_sizes, self.lock = [], [], threading.Lock()

    def create(self, model, messages, **params):
        system, user = messages[0]["content"], messages[1]["content"]
        if "Classification Agent" in system:
            kind, out = "classify", {"document_type": "business_process", "confidence": 0.9,
                                     "reason": "test", "recommended_graph": "steps"}
        elif "Planning Agent" in system:
            kind, out = "plan", {"document_title": "Test", "objective": "o", "actors_or_entities": [],
                                 "key_terms": [], "extraction_guidance": "g"}
        elif "Extraction Agent" in system:
            kind, out = "extract", self.extract(user)
        else:
            kind, out = "synthesize", self.synthesize(user, "graph_title" in system)
        content = json.dumps(out)
        with self.lock:
            self.calls[kind] += 1
            if kind == "synthesize":
                self.reply_sizes.append(len(content))
        return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])

    def extract(self, user):
        number = int(user.split("CHUNK NUMBER:")[1].split()[0])
        if number in self.fail_on_chunks:
            raise RuntimeError(f"simulated outage on chunk {number}")
        with self.lock:
            self.extracted.append(number)
        excerpt = user.split("DOCUMENT EXCERPT:")[1]
        sentences = [s.strip() for s in re.split(r"[.\n]", excerpt) if len(s.strip()) > 15]
        nodes = [{"id": f"local-{i}", "name": s[:60], "description": s, "type": "activity",
                  "actor": None, "evidence": [s]} for i, s in enumerate(sentences, 1)]
        edges = [{"from": a["id"], "to": b["id"], "label": None, "evidence": []}
                 for a, b in zip(nodes, nodes[1:], strict=False)]
        return {"nodes": nodes, "edges": edges, "notes": []}

    def synthesize(self, user, final):
        listing = user.split("NODES (id | type | name | actor | description):")[1].split("\n\nEDGES:")[0]
        rows = [line.split(" | ") for line in listing.strip().splitlines()]
        by_name, positions = {}, {}
        for row in rows:
            node_id, name = row[0], row[2]
            by_name.setdefault(name, []).append(node_id)
            chunk, local = re.match(r"C(\d+):local-(\d+)", node_id).groups()
            positions.setdefault(int(chunk), []).append((int(local), node_id))
        merges = [{"ids": ids, "name": name} for name, ids in by_name.items() if len(ids) > 1]
        chunks = sorted(positions)
        new_edges = [{"from": max(positions[a])[1], "to": min(positions[b])[1], "label": "next",
                      "evidence": ["handoff"]} for a, b in zip(chunks, chunks[1:], strict=False)]
        out = {"merges": merges, "new_edges": new_edges, "open_questions": []}
        if final:
            out.update(graph_title="Test graph", description="d", assumptions=[])
        return out
