"""Run (start, resume, reuse or continue after review) the workflow for one document.

Shared by main.py (command line) and evaluate.py (evaluation loop).
"""
import hashlib
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from core.workflow import build_workflow


def default_thread_id(document_path, model, chunk_size, chunk_overlap, merge_batch_size):
    """Same document + same settings -> same run id, so the run can be resumed."""
    digest = hashlib.sha256(Path(document_path).read_bytes())
    digest.update(f"{model}|{chunk_size}|{chunk_overlap}|{merge_batch_size}".encode())
    return f"{Path(document_path).stem}-{digest.hexdigest()[:12]}"


SUPPORTED_TYPES = {"business_process", "requirements", "generic_knowledge"}


def _check_edit(stage, edited):
    """Reject an edited file that doesn't fit the paused stage (e.g. a plan file
    given while the graph is under review)."""
    if edited is None:
        return
    if not isinstance(edited, dict):
        raise ValueError("The edited file must contain a JSON object.")
    if stage == "plan":
        if "plan" not in edited and "classification" not in edited:
            raise ValueError("This run is paused for PLAN review; the edited file needs "
                             "'classification' and/or 'plan'.")
        kind = (edited.get("classification") or {}).get("document_type")
        if kind is not None and kind not in SUPPORTED_TYPES:
            raise ValueError(f"document_type must be one of {sorted(SUPPORTED_TYPES)}, got {kind!r}.")
    if stage == "graph":
        graph = edited.get("graph", edited)
        if not isinstance(graph.get("nodes"), list):
            raise ValueError("This run is paused for GRAPH review; the edited file needs "
                             "a 'graph' with a 'nodes' list.")


def _pending_review(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            return item.value
    return None


def process_document(document_path, client, *, model, output_dir="output", checkpoint_db=None,
                     chunk_size=12000, chunk_overlap=800, merge_batch_size=8, concurrency=4,
                     thread_id=None, fresh=False, review=False, decision=None, log=print):
    """Returns {"status": "done", "graph_file", "thread_id", "reused"}
    or {"status": "review", "review": <what to review>, "thread_id"}.

    decision: None (just report a pending review) or {"edited": <dict or None>}
              to continue a run that is paused for review.
    """
    document_path, output_dir = Path(document_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_db = Path(checkpoint_db or output_dir / "checkpoints.sqlite")
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)

    if not thread_id:
        base = default_thread_id(document_path, model, chunk_size, chunk_overlap, merge_batch_size)
        latest = output_dir / "work" / f"{base}.latest"  # points at the newest --fresh run
        if fresh:
            thread_id = f"{base}-{int(time.time())}"
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text(thread_id)
        else:  # later commands (--approve, --edited, resume) continue the newest run
            thread_id = latest.read_text().strip() if latest.exists() else base
    elif fresh:
        thread_id += f"-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id},
              "max_concurrency": concurrency, "recursion_limit": 100}

    with SqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
        app = build_workflow(client, model, checkpointer, merge_batch_size,
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                             concurrency=concurrency, progress=log)
        snapshot = app.get_state(config)
        pending = _pending_review(snapshot)
        reused = False

        if pending and decision is not None:
            _check_edit(pending["stage"], decision.get("edited"))
            log(f"Continuing after the {pending['stage']} review.")
            app.invoke(Command(resume=decision), config=config)
        elif pending:
            pass  # still waiting for a reviewer
        elif snapshot.next:
            log(f"Resuming unfinished run {thread_id} at step: {', '.join(sorted(set(snapshot.next)))}")
            app.invoke(None, config=config)
        elif snapshot.values.get("graph_file"):
            log(f"Reusing finished run {thread_id}: no LLM calls needed (add --fresh to recompute).")
            reused = True
        else:
            app.invoke({
                "review": review,
                "document_name": document_path.name,
                "document_path": str(document_path),
                "work_dir": str(output_dir / "work" / thread_id),
            }, config=config)

        snapshot = app.get_state(config)
        pending = _pending_review(snapshot)
        if pending:
            return {"status": "review", "review": pending, "thread_id": thread_id}
        return {"status": "done", "graph_file": snapshot.values["graph_file"], "thread_id": thread_id,
                "reused": reused}
