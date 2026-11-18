"""LangGraph flow:

START -> prepare -> classify -> plan -> [review_plan] -> extract (parallel, Send)
      -> synthesize (one reduction level per step, loops) -> validate
      -> [review_graph] -> END

[review_*] pause the run for a human only when the run was started with
review=True (LangGraph interrupt); otherwise they pass straight through.

Large data (document text, chunks, fragments, intermediate graphs) is written
to a work folder; the checkpointed state holds only file paths and small
values, so the SQLite checkpoint stays small. Every step is checkpointed, so a
failed run resumes where it stopped.
"""
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send, interrupt

from agents.classifier_agent import ClassifierAgent
from agents.extraction_agent import ExtractionAgent
from agents.planner_agent import PlannerAgent
from agents.synthesis_agent import SynthesisAgent
from core.chunker import build_preview, split_into_chunks
from core.document_reader import read_document
from core.evaluation import evidence_grounding
from core.graph_utils import finalize, normalize_fragment, validate_graph

SUPPORTED_TYPES = {"business_process", "requirements", "generic_knowledge"}


class State(TypedDict, total=False):
    review: bool
    document_name: str
    document_path: str
    work_dir: str
    chunk_count: int
    classification: dict
    document_type: str
    plan: dict
    partial_results: Annotated[list[dict], operator.add]  # {"chunk": n, "file": path}
    level: int
    level_files: list[str]
    synthesis_done: bool
    graph_file: str


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)  # atomic: a crash never leaves a half-written file
    return str(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_workflow(client, model, checkpointer, batch_size=8,
                   chunk_size=12000, chunk_overlap=800, concurrency=4, progress=None):
    """progress: optional callable(str) that receives readable status lines."""
    say = progress or (lambda message: None)
    classifier = ClassifierAgent(client, model)
    planner = PlannerAgent(client, model)
    extractor = ExtractionAgent(client, model)
    synthesizer = SynthesisAgent(client, model, batch_size, concurrency)

    def prepare(state):
        text = read_document(state["document_path"])
        chunks = split_into_chunks(text, chunk_size, chunk_overlap)
        work = Path(state["work_dir"])
        for number, chunk in enumerate(chunks, 1):
            (work / "chunks").mkdir(parents=True, exist_ok=True)
            (work / "chunks" / f"chunk_{number}.txt").write_text(chunk, encoding="utf-8")
        (work / "preview.txt").write_text(build_preview(chunks), encoding="utf-8")
        say(f"Read the document: {len(text):,} characters, split into "
            f"{len(chunks)} chunk{'' if len(chunks) == 1 else 's'}.")
        return {"chunk_count": len(chunks)}

    def preview(state):
        return (Path(state["work_dir"]) / "preview.txt").read_text(encoding="utf-8")

    def classify(state):
        classification = classifier.run(state["document_name"], preview(state))
        document_type = classification.get("document_type", "generic_knowledge")
        if document_type not in SUPPORTED_TYPES:
            document_type = "generic_knowledge"
        classification["document_type"] = document_type
        confidence = classification.get("confidence")
        say(f"Document type: {document_type}"
            + (f" (confidence {confidence})" if confidence is not None else "") + ".")
        return {"classification": classification, "document_type": document_type}

    def plan(state):
        result = planner.run(state["classification"], preview(state))
        say(f"Extraction plan ready: {len(result.get('actors_or_entities') or [])} actors/entities, "
            f"{len(result.get('key_terms') or [])} key terms.")
        return {"plan": result}

    def fanout(state):
        count = state["chunk_count"]
        say(f"Extracting {count} chunk{'' if count == 1 else 's'} "
            f"(up to {concurrency} at a time)...")
        return [
            Send("extract", {
                "document_type": state["document_type"],
                "plan": state["plan"],
                "work_dir": state["work_dir"],
                "number": number,
                "total": state["chunk_count"],
            })
            for number in range(1, state["chunk_count"] + 1)
        ]

    def extract(state):
        work = Path(state["work_dir"])
        chunk = (work / "chunks" / f"chunk_{state['number']}.txt").read_text(encoding="utf-8")
        fragment = extractor.run(state["document_type"], state["plan"], chunk, state["number"])
        path = save_json(work / "fragments" / f"chunk_{state['number']}.json", fragment)
        say(f"  chunk {state['number']}/{state.get('total', '?')}: "
            f"{len(fragment.get('nodes') or [])} steps, {len(fragment.get('edges') or [])} links")
        return {"partial_results": [{"chunk": state["number"], "file": path}]}

    def synthesize(state):
        """One reduction level per step, so each level is checkpointed."""
        work = Path(state["work_dir"])
        if state.get("level_files"):
            graphs = [load_json(f) for f in state["level_files"]]
        else:  # first level: normalized fragments in document order
            refs = sorted(state.get("partial_results", []), key=lambda r: r["chunk"])
            graphs = [normalize_fragment(load_json(r["file"]), r["chunk"]) for r in refs]
        level = state.get("level", 0) + 1
        steps_before = sum(len(g["nodes"]) for g in graphs)
        final = len(graphs) <= synthesizer.batch_size
        say(f"Merging, level {level}: {len(graphs)} part{'' if len(graphs) == 1 else 's'}, "
            f"{steps_before} steps" + (" (final pass)..." if final else "..."))
        graphs, done = synthesizer.reduce_level(state["document_type"], state["plan"], graphs)
        steps_after = sum(len(g["nodes"]) for g in graphs)
        say(f"  merged {steps_before - steps_after} duplicate step(s); "
            f"{steps_after} steps, {sum(len(g['edges']) for g in graphs)} links remain.")
        files = [save_json(work / "levels" / f"level_{level}_{i}.json", g) for i, g in enumerate(graphs)]
        return {"level": level, "level_files": files, "synthesis_done": done}

    def after_synthesis(state):
        return "validate" if state["synthesis_done"] else "synthesize"

    def source_text(state):
        chunks = sorted((Path(state["work_dir"]) / "chunks").glob("chunk_*.txt"))
        return "\n".join(c.read_text(encoding="utf-8") for c in chunks)

    def check(graph, state):
        """Structural validation + evidence grounding (quotes found in the document)."""
        graph = validate_graph(graph, state["document_type"])
        grounding = evidence_grounding(graph, source_text(state))
        graph["grounding"] = grounding
        missing = grounding["quotes_checked"] - grounding["quotes_grounded"]
        if missing:
            graph["validation_issues"].append(
                f"{missing} evidence quote{'' if missing == 1 else 's'} not found in the document "
                "(reworded or possibly invented); see grounding.ungrounded_examples."
            )
        return graph

    def validate(state):
        graph = finalize(load_json(state["level_files"][0]))
        graph["source_document"] = state["document_name"]
        graph["generated_at"] = datetime.now().isoformat(timespec="seconds")
        graph["document_type"] = state["document_type"]
        graph["classification"] = state["classification"]
        graph["plan"] = state["plan"]
        graph = check(graph, state)
        grounding = graph["grounding"]
        say(f"Checked the graph: {len(graph['validation_issues'])} issue(s); "
            f"{grounding['quotes_grounded']} of {grounding['quotes_checked']} evidence quotes found in the document.")
        return {"graph_file": save_json(Path(state["work_dir"]) / "final_graph.json", graph)}

    def review_plan(state):
        """Human checkpoint 1: graph type + extraction plan, before any extraction spend."""
        if not state.get("review"):
            return {}
        reply = interrupt({"stage": "plan", "classification": state["classification"], "plan": state["plan"]})
        edited = (reply or {}).get("edited")
        if not edited:
            return {}
        classification = edited.get("classification") or state["classification"]
        document_type = classification.get("document_type")
        if document_type not in SUPPORTED_TYPES:
            document_type = state["document_type"]
        classification["document_type"] = document_type
        return {"classification": classification, "document_type": document_type,
                "plan": edited.get("plan") or state["plan"]}

    def review_graph(state):
        """Human checkpoint 2: the finished graph (flagged items listed first by main.py)."""
        if not state.get("review"):
            return {}
        reply = interrupt({"stage": "graph", "graph_file": state["graph_file"]})
        edited = (reply or {}).get("edited")
        if not edited:
            return {}
        graph = edited.get("graph", edited)  # accept the review file or a bare graph
        original = load_json(state["graph_file"])
        for key in ("document_type", "classification", "plan", "source_document", "generated_at",
                    "graph_title", "description", "assumptions"):
            if key in original:
                graph.setdefault(key, original[key])
        graph = check(graph, state)
        graph["reviewed"] = True
        return {"graph_file": save_json(Path(state["work_dir"]) / "final_graph_reviewed.json", graph)}

    retry = RetryPolicy(max_attempts=3)
    workflow = StateGraph(State)
    workflow.add_node("prepare", prepare)
    workflow.add_node("classify", classify, retry_policy=retry)
    workflow.add_node("plan", plan, retry_policy=retry)
    workflow.add_node("extract", extract, retry_policy=retry)
    workflow.add_node("synthesize", synthesize, retry_policy=retry)
    workflow.add_node("validate", validate)
    workflow.add_node("review_plan", review_plan)
    workflow.add_node("review_graph", review_graph)

    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "classify")
    workflow.add_edge("classify", "plan")
    workflow.add_edge("plan", "review_plan")
    workflow.add_conditional_edges("review_plan", fanout, ["extract"])
    workflow.add_edge("extract", "synthesize")
    workflow.add_conditional_edges("synthesize", after_synthesis, ["synthesize", "validate"])
    workflow.add_edge("validate", "review_graph")
    workflow.add_edge("review_graph", END)

    return workflow.compile(checkpointer=checkpointer)
