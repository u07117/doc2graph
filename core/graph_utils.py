"""Deterministic graph operations (no LLM).

The LLM decides WHAT to merge and which cross-fragment edges to add; this
module applies those decisions. Nodes are never dropped, so synthesis is
lossless, and LLM replies stay small no matter how large the graph gets.
"""
import json
from collections import deque


def _text(value, default=""):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _evidence(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [_text(v) for v in value if v not in (None, "")]
    return [_text(value)]


def _unique(items):
    return list(dict.fromkeys(items))


def _node(raw, node_id, order, chunks):
    return {
        "id": node_id,
        "name": _text(raw.get("name")).strip() or node_id,
        "description": _text(raw.get("description")),
        "type": (_text(raw.get("type")).strip().lower() or "other"),
        "actor": _text(raw.get("actor")).strip() or "Unknown",
        "evidence": _evidence(raw.get("evidence")),
        "source_chunks": chunks,
        "order": order,
    }


def normalize_fragment(fragment, chunk_number):
    """Extraction output -> graph with globally unique ids ("C3:local-1").

    Every chunk uses local ids like local-1, so they must be made unique
    before fragments are combined.
    """
    nodes, local_to_global = [], {}
    for index, raw in enumerate(fragment.get("nodes") or []):
        if not isinstance(raw, dict):
            continue
        local = _text(raw.get("id")).strip() or f"auto-{index}"
        if local in local_to_global:
            continue
        global_id = f"C{chunk_number}:{local}"
        local_to_global[local] = global_id
        nodes.append(_node(raw, global_id, chunk_number * 10000 + index, [chunk_number]))

    edges, notes = [], []
    for raw in fragment.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        source = local_to_global.get(_text(raw.get("from")).strip())
        target = local_to_global.get(_text(raw.get("to")).strip())
        if source and target and source != target:
            edges.append({
                "from": source,
                "to": target,
                "label": _text(raw.get("label")),
                "evidence": _evidence(raw.get("evidence")),
            })
        else:
            notes.append(f"(chunk {chunk_number}) dropped edge with unknown endpoint: {raw}")

    notes += [f"(chunk {chunk_number}) {_text(n)}" for n in fragment.get("notes") or [] if n]
    return {"nodes": nodes, "edges": edges, "notes": notes, "open_questions": []}


def combine(graphs):
    return {
        "nodes": [n for g in graphs for n in g["nodes"]],
        "edges": [e for g in graphs for e in g["edges"]],
        "notes": _unique(n for g in graphs for n in g.get("notes", [])),
        "open_questions": _unique(q for g in graphs for q in g.get("open_questions", [])),
    }


def compact_listing(graph, max_notes=150):
    """Compact text view of a graph for the synthesis prompt (~1 line per node)."""
    nodes = "\n".join(
        f"{n['id']} | {n['type']} | {n['name']} | {n['actor']} | {n['description'][:160]}"
        for n in sorted(graph["nodes"], key=lambda n: n["order"])
    )
    edges = "\n".join(
        f"{e['from']} -> {e['to']}" + (f" [{e['label']}]" if e["label"] else "")
        for e in graph["edges"]
    )
    notes = "\n".join(f"- {n[:300]}" for n in graph["notes"][-max_notes:])
    return (
        f"NODES (id | type | name | actor | description):\n{nodes or '(none)'}\n\n"
        f"EDGES:\n{edges or '(none)'}\n\n"
        f"NOTES FROM EXTRACTION (cross-references, ambiguities):\n{notes or '(none)'}"
    )


def apply_synthesis(graph, result):
    """Apply an LLM synthesis decision: merge groups + new edges + questions."""
    by_id = {n["id"]: n for n in graph["nodes"]}
    alias, used = {}, set()

    for group in result.get("merges") or []:
        if not isinstance(group, dict):
            continue
        ids = [i for i in _unique(_text(x) for x in group.get("ids") or [])
               if i in by_id and i not in used]
        if not ids:
            continue  # a single id is allowed: it corrects that node (e.g. type -> start)
        used.update(ids)
        keep = by_id[ids[0]]
        for other_id in ids[1:]:
            other = by_id[other_id]
            alias[other_id] = keep["id"]
            keep["evidence"] = _unique(keep["evidence"] + other["evidence"])
            keep["source_chunks"] = sorted(set(keep["source_chunks"]) | set(other["source_chunks"]))
            keep["order"] = min(keep["order"], other["order"])
            if keep["actor"] == "Unknown":
                keep["actor"] = other["actor"]
        for field in ("name", "description", "type", "actor"):
            if _text(group.get(field)).strip():
                keep[field] = _text(group[field]).strip()
        keep["type"] = keep["type"].lower()

    nodes = [n for n in graph["nodes"] if n["id"] not in alias]
    valid = {n["id"] for n in nodes}
    edges, index = [], {}

    def add_edge(source, target, label, evidence, synthesized=False):
        source, target = alias.get(source, source), alias.get(target, target)
        if source not in valid or target not in valid or source == target:
            return False
        key = (source, target, label.strip().lower())
        if key in index:
            index[key]["evidence"] = _unique(index[key]["evidence"] + evidence)
            return False
        edge = {"from": source, "to": target, "label": label, "evidence": evidence}
        if synthesized:
            edge["added_in_synthesis"] = True
        index[key] = edge
        edges.append(edge)
        return True

    for e in graph["edges"]:
        add_edge(e["from"], e["to"], e["label"], e["evidence"], e.get("added_in_synthesis", False))
    for raw in result.get("new_edges") or []:
        if isinstance(raw, dict):
            add_edge(_text(raw.get("from")), _text(raw.get("to")), _text(raw.get("label")),
                     _evidence(raw.get("evidence")), synthesized=True)

    questions = [_text(q) for q in result.get("open_questions") or [] if q]
    return {
        "nodes": nodes,
        "edges": edges,
        "notes": graph["notes"],
        "open_questions": _unique(graph["open_questions"] + questions),
    }


def finalize(graph):
    """Renumber nodes N1..Nk in document order and drop internal fields."""
    nodes = sorted(graph["nodes"], key=lambda n: n["order"])
    new_ids = {n["id"]: f"N{i}" for i, n in enumerate(nodes, 1)}
    return {
        **{k: v for k, v in graph.items() if k not in ("nodes", "edges", "notes")},
        "nodes": [{**{k: v for k, v in n.items() if k != "order"}, "id": new_ids[n["id"]]} for n in nodes],
        "edges": [{**e, "from": new_ids[e["from"]], "to": new_ids[e["to"]]} for e in graph["edges"]],
    }


def validate_graph(g, document_type=None):
    """Clean the graph defensively and record structural problems.

    Problems are reported in g["validation_issues"] (not auto-fixed), so a
    fragmented or dead-ended graph never passes silently.
    """
    for key in ("nodes", "edges", "assumptions", "open_questions"):
        if not isinstance(g.get(key), list):
            g[key] = []

    ids, nodes = set(), []
    for index, raw in enumerate(g["nodes"]):
        if not isinstance(raw, dict):
            continue
        node_id = _text(raw.get("id")).strip()
        if not node_id or node_id in ids:
            g["open_questions"].append(f"Missing/duplicate node ID excluded: {node_id or '(none)'}")
            continue
        ids.add(node_id)
        node = _node(raw, node_id, index, raw.get("source_chunks") or [])
        node.pop("order")
        nodes.append({**raw, **node})

    edges = []
    for raw in g["edges"]:
        if not isinstance(raw, dict):
            continue
        source, target = _text(raw.get("from")).strip(), _text(raw.get("to")).strip()
        if source in ids and target in ids:
            edges.append({**raw, "from": source, "to": target,
                          "label": _text(raw.get("label")), "evidence": _evidence(raw.get("evidence"))})
        else:
            g["open_questions"].append(f"Edge with unknown endpoint excluded: {raw}")

    g["nodes"], g["edges"] = nodes, edges
    g["validation_issues"] = structural_issues(nodes, edges, document_type or g.get("document_type"))
    return g


def structural_issues(nodes, edges, document_type):
    issues = []
    if not nodes:
        return ["The graph is empty: nothing was extracted from the document."]

    out_edges = {n["id"]: [] for n in nodes}
    in_degree = {n["id"]: 0 for n in nodes}
    neighbours = {n["id"]: set() for n in nodes}
    for e in edges:
        out_edges[e["from"]].append(e)
        in_degree[e["to"]] += 1
        neighbours[e["from"]].add(e["to"])
        neighbours[e["to"]].add(e["from"])

    isolated = [n["id"] for n in nodes if not neighbours[n["id"]]]
    if isolated:
        issues.append(f"{len(isolated)} step(s) are not linked to anything: {', '.join(isolated[:20])}.")

    seen, components = set(), 0
    for n in nodes:
        if n["id"] in seen or not neighbours[n["id"]]:
            continue
        components += 1
        queue = deque([n["id"]])
        while queue:
            current = queue.popleft()
            if current not in seen:
                seen.add(current)
                queue.extend(neighbours[current] - seen)
    if components > 1:
        issues.append(f"The graph is split into {components} disconnected parts; some links between them are missing.")

    if document_type == "business_process":
        types = {n["id"]: n["type"] for n in nodes}
        starts = [i for i, t in types.items() if t == "start"]
        if not starts:
            issues.append("No start node: no step is marked as where the process begins.")
        if not any(t == "end" for t in types.values()):
            issues.append("No end node: no step is marked as a final outcome.")
        for n in nodes:
            node_id, kind = n["id"], n["type"]
            if kind != "end" and neighbours[node_id] and not out_edges[node_id]:
                issues.append(f"{node_id} '{n['name']}' is a dead end: nothing follows it, but it isn't marked as an end.")
            if kind != "start" and neighbours[node_id] and not in_degree[node_id]:
                issues.append(f"{node_id} '{n['name']}' has no incoming link; only the start step should have none.")
            if kind == "decision" and len(out_edges[node_id]) < 2:
                count = len(out_edges[node_id])
                issues.append(f"Decision {node_id} '{n['name']}' has only {count} outgoing branch"
                              f"{'' if count == 1 else 'es'}; a decision needs at least 2.")
        if starts:
            reached, queue = set(starts), deque(starts)
            while queue:
                for e in out_edges[queue.popleft()]:
                    if e["to"] not in reached:
                        reached.add(e["to"])
                        queue.append(e["to"])
            unreachable = [n["id"] for n in nodes if n["id"] not in reached and neighbours[n["id"]]]
            if unreachable:
                issues.append(f"{len(unreachable)} step(s) can't be reached from the start: {', '.join(unreachable[:20])}.")
    return issues
