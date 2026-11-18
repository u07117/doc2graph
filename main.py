import argparse
import json
import logging
import sys
import time
from pathlib import Path

from agents.base_agent import BaseAgent
from core.document_reader import read_document
from core.html_renderer import render_html
from core.llm_client import create_client, default_model
from core.runner import process_document
from core.workflow import load_json

try:  # keys and models can live in a .env file (see .env.example)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a document and generate its graph.")
    parser.add_argument("document", nargs="?", default="sample_documents/loan_approval_policy.md",
                        help="Input PDF, DOCX, TXT or Markdown document.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--checkpoint-db", default=None, help="Default: <output-dir>/checkpoints.sqlite")
    parser.add_argument("--thread-id", default=None,
                        help="Run id. Default: derived from the document and settings, so re-running "
                        "the same command resumes or reuses the same run.")
    parser.add_argument("--fresh", action="store_true", help="Start a new run instead of resuming/reusing.")
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--chunk-size", type=int, default=12000)
    parser.add_argument("--chunk-overlap", type=int, default=800)
    parser.add_argument("--merge-batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel LLM calls (lower it if you hit rate limits).")
    review = parser.add_argument_group("human review")
    review.add_argument("--review", action="store_true",
                        help="Pause for review after the plan and after the graph is built.")
    review.add_argument("--approve", action="store_true", help="Continue a paused run as-is.")
    review.add_argument("--edited", metavar="FILE", help="Continue a paused run with your edited review file.")
    return parser.parse_args()


def graph_review_file(graph):
    """What a reviewer should look at first, then the full graph to edit."""
    names = {n["id"]: n["name"] for n in graph["nodes"]}
    added = [f"{e['from']} '{names[e['from']]}' -> {e['to']} '{names[e['to']]}'"
             + (f" [{e['label']}]" if e.get("label") else "")
             for e in graph["edges"] if e.get("added_in_synthesis")]
    decisions = []
    for node in graph["nodes"]:
        if node.get("type") == "decision":
            exits = [f"-> {e['to']} [{e.get('label') or 'no label'}]" for e in graph["edges"] if e["from"] == node["id"]]
            decisions.append(f"{node['id']} '{node['name']}': " + (", ".join(exits) or "NO EXITS"))
    return {
        "how_to_review": "Check 'review_first', then fix anything in 'graph' (nodes/edges) and continue with "
                         "--edited <this file>, or continue unchanged with --approve.",
        "review_first": {
            "validation_issues": graph.get("validation_issues", []),
            "open_questions": graph.get("open_questions", []),
            "ungrounded_evidence": graph.get("grounding", {}).get("ungrounded_examples", []),
            "links_added_in_synthesis": added,
            "decisions_and_their_branches": decisions,
        },
        "graph": graph,
    }


def _plural(count, word, suffix="s"):
    return f"{count} {word}{'' if count == 1 else suffix}"


def print_pause(pending, review_file, command, graph=None):
    if pending["stage"] == "plan":
        kind = pending["classification"].get("document_type")
        confidence = pending["classification"].get("confidence")
        print("\nPaused for review (1 of 2): graph type and extraction plan")
        print(f"  Detected type:  {kind}" + (f" (confidence {confidence})" if confidence is not None else ""))
    else:
        first = graph_review_file(graph)["review_first"]
        print("\nPaused for review (2 of 2): the finished graph")
        print(f"  Size:           {_plural(len(graph['nodes']), 'step')}, {_plural(len(graph['edges']), 'link')}")
        print("  Look first at:  "
              f"{_plural(len(first['validation_issues']), 'issue')} · "
              f"{_plural(len(first['ungrounded_evidence']), 'quote')} not found · "
              f"{_plural(len(first['links_added_in_synthesis']), 'link')} added across chunks · "
              f"{_plural(len(first['decisions_and_their_branches']), 'decision')}")
    print(f"  Review file:    {review_file}")
    print("  Then run one of:")
    print("    continue unchanged:")
    print(f"      {command} --approve")
    print("    continue with your edits:")
    print(f"      {command} --edited {review_file}")


def print_summary(graph, result, json_path, html_path, elapsed, calls):
    nodes, edges = graph.get("nodes", []), graph.get("edges", [])
    decisions = sum(1 for n in nodes if n.get("type") == "decision")
    grounding = graph.get("grounding", {})
    checked, found = grounding.get("quotes_checked", 0), grounding.get("quotes_grounded", 0)
    issues, questions = graph.get("validation_issues", []), graph.get("open_questions", [])
    classification = graph.get("classification") or {}

    print()
    if result.get("reused"):
        print("Done · reused the finished run, no LLM calls")
    else:
        print(f"Done in {elapsed:.0f}s · " + (_plural(calls, "LLM call") if calls else "no LLM calls needed"))
    print(f"\nGraph: {graph.get('graph_title') or 'Untitled'}")
    confidence = classification.get("confidence")
    print(f"  Type       {graph.get('document_type')}"
          + (f" (confidence {confidence})" if confidence is not None else ""))
    print(f"  Size       {_plural(len(nodes), 'step')} · {_plural(len(edges), 'link')} · "
          f"{_plural(decisions, 'decision')}")
    if checked:
        print(f"  Evidence   {found} of {checked} quotes found in the document ({found / checked:.0%})")
    print("  Review     " + ("human-reviewed" if graph.get("reviewed") else
                             "not reviewed (add --review to check it before it's final)"))

    if issues:
        print(f"\nNeeds attention: {_plural(len(issues), 'issue')}")
        for number, issue in enumerate(issues[:10], 1):
            print(f"  {number}. {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more in the JSON file")
        for example in grounding.get("ungrounded_examples", [])[:3]:
            print(f"     not found ({example['where']}): \"{example['quote'][:100]}\"")
    else:
        print("\nNeeds attention: nothing, all checks passed")

    if questions:
        print(f"\nOpen questions: {len(questions)}")
        for question in questions[:5]:
            print(f"  - {question}")
        if len(questions) > 5:
            print(f"  ... and {len(questions) - 5} more in the JSON file")

    print("\nOutputs")
    print(f"  Diagram    {html_path}   <- open this")
    print(f"  Data       {json_path}")
    print(f"  Run id     {result['thread_id']}")


def main():
    args = parse_args()
    started = time.time()

    def log(message):
        print(f"[{time.time() - started:5.0f}s] {message}", flush=True)

    logging.basicConfig(level=logging.WARNING, format="        ! %(message)s")
    document_path, output_dir = Path(args.document), Path(args.output_dir)

    if not document_path.is_file():
        raise SystemExit(f"Document not found: {document_path}")
    try:
        client = create_client()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    try:
        if not read_document(document_path).strip():
            raise SystemExit("No extractable text found (scanned PDFs need OCR).")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    decision = None
    if args.approve or args.edited:
        edited = None
        if args.edited:
            try:
                edited = json.loads(Path(args.edited).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Cannot read --edited file: {exc}") from None
        decision = {"edited": edited}

    print(f"Document-to-Graph · {document_path.name} · model {args.model}"
          + (" · review on" if args.review else ""))
    calls_before = BaseAgent.calls_made
    try:
        result = process_document(
            document_path, client, model=args.model, output_dir=output_dir,
            checkpoint_db=args.checkpoint_db, chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap, merge_batch_size=args.merge_batch_size,
            concurrency=args.concurrency, thread_id=args.thread_id, fresh=args.fresh,
            review=args.review, decision=decision, log=log,
        )
    except ValueError as exc:  # an edited file that doesn't fit the paused stage
        raise SystemExit(f"Review edit rejected: {exc}") from None
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Progress is saved; run the same command again to resume.")
    except Exception as exc:
        sys.exit(f"\nRun failed: {type(exc).__name__}: {exc}\n"
                 "Progress is saved; run the same command again to resume.")

    review_dir = output_dir / "review" / result["thread_id"]
    current_stage = result["review"]["stage"] if result["status"] == "review" else None
    for old in review_dir.glob("*.json"):  # review files of stages already passed
        if old.stem != current_stage:
            old.unlink()

    command = f"python main.py {document_path}"
    if result["status"] == "review":
        pending = result["review"]
        review_file = review_dir / f"{pending['stage']}.json"
        if not review_file.exists():  # never overwrite: it may already hold your edits
            if pending["stage"] == "plan":
                content = {"how_to_review": "Check the graph type and the plan, edit if needed, then continue "
                                            "with --edited <this file>, or continue unchanged with --approve.",
                           "classification": pending["classification"], "plan": pending["plan"]}
            else:
                content = graph_review_file(load_json(pending["graph_file"]))
            review_dir.mkdir(parents=True, exist_ok=True)
            review_file.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        graph = load_json(pending["graph_file"]) if pending["stage"] == "graph" else None
        print_pause(pending, review_file, command, graph)
        return

    graph = load_json(result["graph_file"])
    json_path, html_path = output_dir / "document_graph.json", output_dir / "document_graph.html"
    json_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(graph), encoding="utf-8")

    print_summary(graph, result, json_path, html_path, time.time() - started,
                  BaseAgent.calls_made - calls_before)


if __name__ == "__main__":
    main()
