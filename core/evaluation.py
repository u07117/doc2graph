"""Quality checks for generated graphs.

1. evidence_grounding(): reference-free. Checks that each evidence quote really
   appears in the source document. Runs on every graph (see core/workflow.py).
2. score(): compares a generated graph with a hand-built reference ("gold")
   graph. Used by evaluate.py.

Matching is lexical (no LLM), so the scores are deterministic and free. A
reference node counts as found when a generated node's name is similar enough
(token overlap or character similarity >= threshold) to its name or to one of
its optional "aliases" (add synonyms there, e.g. "Start the funds transfer").
"""
import re
from difflib import SequenceMatcher

SHINGLE = 5  # words per shingle for fuzzy quote matching


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def similarity(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    tokens_a, tokens_b = set(a.split()), set(b.split())
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return max(jaccard, SequenceMatcher(None, a, b).ratio())


def _ratio(part, whole):
    return round(part / whole, 3) if whole else None


def _f1(p, r):
    return round(2 * p * r / (p + r), 3) if p and r else 0.0


# ------------------------------------------------------------------ grounding
def evidence_grounding(graph, source_text, max_examples=20):
    """Share of evidence quotes found in the source (exact after normalizing,
    or at least 80% of their 5-word shingles present)."""
    source = _norm(source_text)
    words = source.split()
    shingles = {" ".join(words[i:i + SHINGLE]) for i in range(max(0, len(words) - SHINGLE + 1))}

    def grounded(quote):
        q = _norm(quote)
        if not q or q in source:
            return True
        qw = q.split()
        if len(qw) < SHINGLE:
            return False
        parts = [" ".join(qw[i:i + SHINGLE]) for i in range(len(qw) - SHINGLE + 1)]
        return sum(p in shingles for p in parts) / len(parts) >= 0.8

    checked, ungrounded = 0, []
    for kind, items in (("node", graph.get("nodes", [])), ("edge", graph.get("edges", []))):
        for item in items:
            for quote in item.get("evidence") or []:
                checked += 1
                if not grounded(quote):
                    where = item.get("id") if kind == "node" else f"{item.get('from')}->{item.get('to')}"
                    ungrounded.append({"where": where, "quote": str(quote)[:200]})
    return {
        "quotes_checked": checked,
        "quotes_grounded": checked - len(ungrounded),
        "rate": _ratio(checked - len(ungrounded), checked),
        "ungrounded_examples": ungrounded[:max_examples],
    }


# ------------------------------------------------------------------ scoring
def match_nodes(pred_nodes, gold_nodes, threshold=0.5):
    """Greedy one-to-one matching by name similarity. Returns {gold_id: pred_id}."""
    def best(gold_node, pred_node):  # reference nodes may list "aliases" (synonyms)
        names = [gold_node.get("name")] + list(gold_node.get("aliases") or [])
        return max(similarity(name, pred_node.get("name")) for name in names)

    pairs = sorted(
        ((best(g, p), g["id"], p["id"]) for g in gold_nodes for p in pred_nodes),
        reverse=True,
    )
    mapping, used = {}, set()
    for score_value, gold_id, pred_id in pairs:
        if score_value < threshold:
            break
        if gold_id not in mapping and pred_id not in used:
            mapping[gold_id] = pred_id
            used.add(pred_id)
    return mapping


def score(pred, gold, source_text=None, threshold=0.5):
    """Compare a generated graph (pred) with a reference graph (gold)."""
    pred_nodes, gold_nodes = pred.get("nodes", []), gold.get("nodes", [])
    pred_by_id = {n["id"]: n for n in pred_nodes}
    gold_by_id = {n["id"]: n for n in gold_nodes}
    mapping = match_nodes(pred_nodes, gold_nodes, threshold)
    pred_to_gold = {p: g for g, p in mapping.items()}

    node_p = _ratio(len(mapping), len(pred_nodes)) or 0.0
    node_r = _ratio(len(mapping), len(gold_nodes)) or 0.0

    gold_edges = {(e["from"], e["to"]): e for e in gold.get("edges", [])}
    found_gold_edges, correct_pred_edges, conditions_checked, conditions_ok = set(), 0, 0, 0
    for e in pred.get("edges", []):
        key = (pred_to_gold.get(e["from"]), pred_to_gold.get(e["to"]))
        if key in gold_edges:
            correct_pred_edges += 1
            if key not in found_gold_edges:
                found_gold_edges.add(key)
                gold_label = gold_edges[key].get("label")
                if gold_label:
                    conditions_checked += 1
                    conditions_ok += similarity(gold_label, e.get("label")) >= threshold
    edge_p = _ratio(correct_pred_edges, len(pred.get("edges", []))) or 0.0
    edge_r = _ratio(len(found_gold_edges), len(gold_edges)) or 0.0

    gold_decisions = [g for g in gold_nodes if g.get("type") == "decision"]
    decisions_found = sum(
        1 for g in gold_decisions
        if g["id"] in mapping and pred_by_id[mapping[g["id"]]].get("type") == "decision"
    )
    actor_pairs = [(gold_by_id[g], pred_by_id[p]) for g, p in mapping.items()
                   if gold_by_id[g].get("actor") not in (None, "", "Unknown")]
    actors_ok = sum(similarity(g["actor"], p.get("actor")) >= 0.6 for g, p in actor_pairs)
    types_ok = sum(gold_by_id[g].get("type") == pred_by_id[p].get("type") for g, p in mapping.items())

    result = {
        "node_precision": node_p,
        "node_recall": node_r,
        "node_f1": _f1(node_p, node_r),
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "edge_f1": _f1(edge_p, edge_r),
        "decision_recall": _ratio(decisions_found, len(gold_decisions)),
        "condition_accuracy": _ratio(conditions_ok, conditions_checked),
        "actor_accuracy": _ratio(actors_ok, len(actor_pairs)),
        "type_accuracy": _ratio(types_ok, len(mapping)),
        "classification_correct": (pred.get("document_type") == gold["document_type"]
                                   if gold.get("document_type") else None),
        "structural_issues": len(pred.get("validation_issues", [])),
        "counts": {"pred_nodes": len(pred_nodes), "gold_nodes": len(gold_nodes),
                   "pred_edges": len(pred.get("edges", [])), "gold_edges": len(gold_edges)},
        "missed_nodes": [g["name"] for g in gold_nodes if g["id"] not in mapping],
        "extra_nodes": [p["name"] for p in pred_nodes if p["id"] not in pred_to_gold],
        "missed_edges": [f"{gold_by_id[a]['name']} -> {gold_by_id[b]['name']}"
                         for (a, b) in gold_edges if (a, b) not in found_gold_edges
                         and a in gold_by_id and b in gold_by_id],
    }
    if source_text is not None:
        result["grounding_rate"] = evidence_grounding(pred, source_text)["rate"]
    return result
