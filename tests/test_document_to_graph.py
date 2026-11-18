import json
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import main as main_module  # noqa: E402
from agents.base_agent import BaseAgent  # noqa: E402
from core.chunker import build_preview, split_into_chunks  # noqa: E402
from core.document_reader import read_document  # noqa: E402
from core.graph_utils import validate_graph  # noqa: E402
from core.html_renderer import mermaid_source, render_html  # noqa: E402
from fake_client import FakeClient  # noqa: E402


def big_document(path, sections=30):
    text = "\n\n".join(
        f"## Section {i}\n" + " ".join(f"The officer completes task {i}-{j} for the file." for j in range(12))
        for i in range(sections)
    )
    path.write_text(text)
    return path


def run_main(tmp_path, doc, client, monkeypatch, *extra):
    monkeypatch.setattr(main_module, "create_client", lambda: client)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "main.py", str(doc), "--output-dir", str(out), "--checkpoint-db", str(out / "cp.sqlite"),
        "--chunk-size", "2000", "--chunk-overlap", "200", "--merge-batch-size", "4", *extra])
    main_module.main()
    return json.loads((out / "document_graph.json").read_text()), out


# ---------------------------------------------------------------- synthesis: lossless + scalable
def test_large_document_is_lossless_linked_and_replies_stay_small(tmp_path, monkeypatch):
    doc, client = big_document(tmp_path / "big.md"), FakeClient()
    graph, out = run_main(tmp_path, doc, client, monkeypatch)

    work = next((out / "work").iterdir())
    fragments = [json.loads(f.read_text()) for f in (work / "fragments").glob("*.json")]
    extracted_names = {n["name"] for f in fragments for n in f["nodes"]}
    assert len(fragments) > 8                                     # many chunks
    assert (work / "levels" / "level_2_0.json").exists()          # multi-level reduction ran
    assert {n["name"] for n in graph["nodes"]} == extracted_names  # nothing lost
    assert len(graph["nodes"]) == len(extracted_names)            # overlap duplicates merged
    assert [n["id"] for n in graph["nodes"]] == [f"N{i}" for i in range(1, len(graph["nodes"]) + 1)]
    assert any(e.get("added_in_synthesis") for e in graph["edges"])
    assert not any("disconnected" in i for i in graph["validation_issues"])  # chunks linked up
    # synthesis replies are decisions, far smaller than the graph they act on
    assert max(client.reply_sizes) < len(json.dumps(graph)) / 5


# ---------------------------------------------------------------- resume / reuse
def test_rerun_reuses_result_without_duplicates(tmp_path, monkeypatch, capsys):
    doc = big_document(tmp_path / "d.md", sections=6)
    first, _ = run_main(tmp_path, doc, FakeClient(), monkeypatch)
    again = FakeClient()
    second, _ = run_main(tmp_path, doc, again, monkeypatch)
    assert sum(again.calls.values()) == 0 and "Reusing finished run" in capsys.readouterr().out
    assert len(second["nodes"]) == len(first["nodes"])           # was doubled before the fix


def test_failed_run_resumes_where_it_stopped(tmp_path, monkeypatch, capsys):
    doc = big_document(tmp_path / "d.md", sections=10)
    bad = FakeClient(fail_on_chunks={3})
    with pytest.raises(SystemExit, match="simulated outage on chunk 3"):
        run_main(tmp_path, doc, bad, monkeypatch)
    done_before = set(bad.extracted)

    good = FakeClient()
    graph, _ = run_main(tmp_path, doc, good, monkeypatch)
    assert "Resuming unfinished run" in capsys.readouterr().out
    assert good.calls["classify"] == 0 and good.calls["plan"] == 0
    assert 3 in good.extracted and not done_before & set(good.extracted)
    assert graph["nodes"]


def test_fresh_starts_a_new_run(tmp_path, monkeypatch):
    doc = big_document(tmp_path / "d.md", sections=4)
    run_main(tmp_path, doc, FakeClient(), monkeypatch)
    again = FakeClient()
    run_main(tmp_path, doc, again, monkeypatch, "--fresh")
    assert again.calls["classify"] == 1


def test_checkpoints_do_not_store_the_document(tmp_path, monkeypatch):
    doc = big_document(tmp_path / "d.md", sections=30)
    _, out = run_main(tmp_path, doc, FakeClient(), monkeypatch)
    db = sqlite3.connect(out / "cp.sqlite")
    blobs = b"".join(r[0] or b"" for r in db.execute("select checkpoint from checkpoints"))
    blobs += b"".join(r[0] or b"" for r in db.execute("select value from writes"))
    assert b"The officer completes task 17-5" not in blobs  # no document text in checkpoints
    assert len(blobs) < 60_000  # small fixed overhead; before the fix it was ~38x the document


# ---------------------------------------------------------------- base agent: retries, fallback, repair
def make_agent(handler):
    from openai import OpenAI

    client = OpenAI(api_key="x", base_url="https://example.test/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return BaseAgent(client, "model-under-test")


def reply(content, finish="stop"):
    return httpx.Response(200, json={"id": "1", "object": "chat.completion", "created": 0, "model": "m",
                                     "choices": [{"index": 0, "finish_reason": finish,
                                                  "message": {"role": "assistant", "content": content}}]})


def test_param_fallback_retry_and_json_repair(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    BaseAgent._dropped_params.clear()
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        if len(bodies) == 2:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if len(bodies) == 3:
            return reply('Sure! ```json\n{"a": 1,}\n```')  # invalid JSON (trailing comma)
        return reply('Here you go: {"a": 1}')

    assert make_agent(handler).ask_json("sys", "user") == {"a": 1}
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert bodies[1]["temperature"] == 0                      # only the rejected param was dropped
    assert "not one valid JSON object" in bodies[3]["messages"][-1]["content"]


def test_gives_up_with_clear_error(monkeypatch):
    BaseAgent._dropped_params.clear()
    with pytest.raises(ValueError, match="failed after 4 attempts"):
        make_agent(lambda r: reply("not json at all")).ask_json("s", "u")


# ---------------------------------------------------------------- rendering + validation
def test_render_survives_nulls_and_awkward_ids():
    graph = validate_graph({
        "nodes": [{"id": "end", "name": None, "type": "end", "actor": None, "description": None, "evidence": None},
                  {"id": "local 1", "name": 'Check "A|B"', "type": "decision", "actor": "Ops"}],
        "edges": [{"from": "local 1", "to": "end", "label": "x|y"}],
    })
    page = render_html(graph)
    source = mermaid_source(graph)
    assert "n0" in source and "n1 -->" in source and "x'y" in source  # safe ids, safe labels
    assert '"A' not in source.split("n1{{")[1].split("}}")[0].strip('"')[:1]  # quotes escaped inside labels
    assert "maxTextSize" in page and "Needs your attention" in page


def test_report_speaks_business_language():
    graph = validate_graph({
        "graph_title": "Loan approval", "document_type": "business_process",
        "description": "This process graph outlines loan approval.",
        "assumptions": ["The process begins with 'Apply' (C1:local-1)."],
        "open_questions": ["The escalation path is not detailed in the excerpt."],
        "nodes": [{"id": "N1", "name": "Apply", "type": "start", "actor": "Applicant", "source_chunks": [1]},
                  {"id": "N2", "name": "Credit Report Availability Decision", "type": "decision", "actor": "Credit Team"},
                  {"id": "N3", "name": "Application On Hold", "type": "event", "actor": None},
                  {"id": "N4", "name": "Loan disbursed", "type": "end", "actor": "Disbursement Team"}],
        "edges": [{"from": "N1", "to": "N2", "label": "Proceed to check"},
                  {"from": "N2", "to": "N3", "label": "No: report unavailable"},
                  {"from": "N2", "to": "N4", "label": "Yes"}],
    }, "business_process")
    import html as html_module

    page = html_module.unescape(render_html(graph))
    visible = page.split("<details class='technical'>")[0]  # what a business reader sees by default
    for internal in ("C1:local-1", "excerpt", "chunks", "nodes", "edges", "business_process"):
        assert internal not in visible.split("<script")[0], internal
    assert "2. Credit Report Availability?" in page            # decisions read as questions
    assert "the document doesn't say what happens next" in page  # dead end phrased as a gap
    assert "who is responsible for step 3" in page             # missing owner
    assert "in the document" in page and "process map outlines" in page
    assert "No: report unavailable" in mermaid_source(graph)    # decision branches labelled
    assert "Proceed to check" not in mermaid_source(graph)      # plain links unlabelled
    assert "4 steps" in page and "1 decision" in page and "1 possible outcome" in page


def test_structural_validation_flags_problems():
    nodes = [{"id": i, "name": i, "type": t} for i, t in
             [("S", "start"), ("D", "decision"), ("A", "activity"), ("X", "activity"), ("Y", "activity")]]
    edges = [{"from": "S", "to": "D"}, {"from": "D", "to": "A"}, {"from": "X", "to": "Y"}]
    issues = " ".join(validate_graph({"nodes": nodes, "edges": edges}, "business_process")["validation_issues"])
    for expected in ("No end node", "dead end", "only 1 outgoing branch", "disconnected", "can't be reached"):
        assert expected in issues


# ---------------------------------------------------------------- input handling
def test_preview_covers_whole_document():
    chunks = [f"PART-{i} " + "x" * 3000 for i in range(60)]
    preview = build_preview(chunks, budget_chars=18000)
    assert "PART-0" in preview and "PART-59" in preview and len(preview) < 19000


def test_no_double_overlap_on_long_paragraphs():
    chunks = split_into_chunks("A" * 5000, max_chars=2000, overlap_chars=200)
    assert all(len(c) <= 2000 + 202 for c in chunks)


def test_docx_tables_are_read_in_order(tmp_path):
    import docx

    d = docx.Document()
    d.add_paragraph("Approval limits:")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Amount", "Approver"
    table.cell(1, 0).text, table.cell(1, 1).text = "Up to 5 lakh", "Branch Manager"
    d.add_paragraph("After approval the offer is issued.")
    d.save(tmp_path / "p.docx")
    text = read_document(tmp_path / "p.docx")
    assert text.index("Approval limits") < text.index("Up to 5 lakh | Branch Manager") < text.index("offer is issued")


def test_non_utf8_text_does_not_crash(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"Caf\xe9 approval step")
    assert "approval step" in read_document(tmp_path / "x.txt")
