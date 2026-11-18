"""Human review flow, evidence grounding, scoring and the evaluation loop."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import evaluate as evaluate_module  # noqa: E402
import main as main_module  # noqa: E402
from core.evaluation import evidence_grounding, score  # noqa: E402
from fake_client import FakeClient  # noqa: E402

GOLD = ROOT / "eval" / "dataset" / "loan_approval_policy.gold.json"
DOC = ROOT / "eval" / "dataset" / "loan_approval_policy.md"


def run_main(tmp_path, monkeypatch, client, *extra):
    monkeypatch.setattr(main_module, "create_client", lambda: client)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["main.py", str(DOC), "--output-dir", str(out), *extra])
    main_module.main()
    return out


def review_file(out, stage):
    files = list((out / "review").glob(f"*/{stage}.json"))
    return files[0] if files else None


# ------------------------------------------------------------------ human review
def test_review_pauses_twice_and_applies_edits(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    out = run_main(tmp_path, monkeypatch, client, "--review")
    assert "Paused for review (1 of 2)" in capsys.readouterr().out
    assert client.calls["extract"] == 0                      # paused before any extraction spend

    plan_file = review_file(out, "plan")
    edited = json.loads(plan_file.read_text())
    edited["classification"]["document_type"] = "requirements"
    plan_file.write_text(json.dumps(edited))

    run_main(tmp_path, monkeypatch, client)                  # re-running while paused...
    assert json.loads(plan_file.read_text())["classification"]["document_type"] == "requirements"  # ...keeps edits

    run_main(tmp_path, monkeypatch, client, "--edited", str(plan_file))
    assert "Paused for review (2 of 2)" in capsys.readouterr().out
    assert not plan_file.exists()                            # consumed review file is removed
    graph_file = review_file(out, "graph")
    review = json.loads(graph_file.read_text())
    assert set(review["review_first"]) >= {"validation_issues", "open_questions", "ungrounded_evidence",
                                           "links_added_in_synthesis", "decisions_and_their_branches"}
    removed = review["graph"]["nodes"][0]["id"]
    review["graph"]["nodes"] = review["graph"]["nodes"][1:]
    graph_file.write_text(json.dumps(review))

    run_main(tmp_path, monkeypatch, client, "--edited", str(graph_file))
    printed = capsys.readouterr().out
    assert "Review     human-reviewed" in printed
    final = json.loads((out / "document_graph.json").read_text())
    assert final["document_type"] == "requirements" and final["reviewed"] is True
    assert removed not in {n["id"] for n in final["nodes"]}
    assert all(removed not in (e["from"], e["to"]) for e in final["edges"])
    assert "Reviewed by a person" in (out / "document_graph.html").read_text()


def test_approve_without_changes(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    run_main(tmp_path, monkeypatch, client, "--review")
    run_main(tmp_path, monkeypatch, client, "--approve")
    run_main(tmp_path, monkeypatch, client, "--approve")
    assert "Done in" in capsys.readouterr().out


def test_bad_edits_are_rejected(tmp_path, monkeypatch):
    client = FakeClient()
    out = run_main(tmp_path, monkeypatch, client, "--review")
    plan_file = review_file(out, "plan")
    bad = json.loads(plan_file.read_text())
    bad["classification"]["document_type"] = "unsupported_type"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="document_type must be one of"):
        run_main(tmp_path, monkeypatch, client, "--edited", str(tmp_path / "bad.json"))
    (tmp_path / "graph_like.json").write_text(json.dumps({"nodes": []}))
    with pytest.raises(SystemExit, match="paused for PLAN review"):
        run_main(tmp_path, monkeypatch, client, "--edited", str(tmp_path / "graph_like.json"))


def test_no_review_runs_straight_through(tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, FakeClient())
    printed = capsys.readouterr().out
    assert "Paused" not in printed and "quotes found in the document" in printed


# ------------------------------------------------------------------ grounding + scoring
def test_grounding_flags_invented_quotes():
    text = DOC.read_text()
    graph = {"nodes": [{"id": "N1", "evidence": ["The Verification Team validates identity documents"]},
                       {"id": "N2", "evidence": ["Loans above 1 crore need board approval"]},
                       {"id": "N3", "evidence": ["the  CREDIT team obtains a credit report!"]}],  # case/space/punct
             "edges": []}
    g = evidence_grounding(graph, text)
    assert g["quotes_checked"] == 3 and g["quotes_grounded"] == 2
    assert g["ungrounded_examples"][0]["where"] == "N2"


def test_score_perfect_and_injected_errors():
    gold = json.loads(GOLD.read_text())
    perfect = score(gold, gold, DOC.read_text())
    assert perfect["node_f1"] == perfect["edge_f1"] == perfect["grounding_rate"] == 1.0

    pred = copy.deepcopy(gold)
    pred["nodes"] = [n for n in pred["nodes"] if n["id"] != "G11"]               # missed step
    pred["edges"] = [e for e in pred["edges"] if "G11" not in (e["from"], e["to"])]
    next(n for n in pred["nodes"] if n["id"] == "G15")["type"] = "activity"      # decision lost
    next(n for n in pred["nodes"] if n["id"] == "G24")["name"] = "Start the funds transfer"  # synonym (alias)
    pred["nodes"].append({"id": "X1", "name": "Invented compliance step", "type": "activity"})
    s = score(pred, gold)
    assert s["node_recall"] < 1 and s["node_precision"] < 1 and s["edge_recall"] < 1
    assert s["decision_recall"] == 0.9
    assert s["missed_nodes"] == ["Application put on hold"]                      # alias matched G24
    assert s["extra_nodes"] == ["Invented compliance step"]


# ------------------------------------------------------------------ evaluation loop
def test_eval_loop_scores_saves_baseline_and_flags_regressions(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / DOC.name).write_text(DOC.read_text())
    (dataset / GOLD.name).write_text(GOLD.read_text())
    monkeypatch.setattr(evaluate_module, "create_client", lambda: FakeClient())

    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(dataset), "--save-baseline"])
    evaluate_module.main()
    printed = capsys.readouterr().out
    assert "AVERAGE" in printed and "Saved baseline" in printed
    baseline = json.loads((tmp_path / "eval" / "baseline.json").read_text())
    assert baseline["averages"]["grounding_rate"] == 1.0
    assert list((tmp_path / "eval" / "results").glob("eval_*.json"))

    # Pretend the baseline was much better -> the same run is now a regression (exit 1).
    baseline["averages"]["node_f1"] = 0.99
    (tmp_path / "eval" / "baseline.json").write_text(json.dumps(baseline))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(dataset)])
    with pytest.raises(SystemExit) as stop:
        evaluate_module.main()
    assert stop.value.code == 1 and "REGRESSION" in capsys.readouterr().out


def test_score_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--score", str(GOLD), str(GOLD), "--document", str(DOC)])
    evaluate_module.main()
    assert json.loads(capsys.readouterr().out)["node_f1"] == 1.0


# ------------------------------------------------------------------ single-node corrections
def test_synthesis_can_correct_a_single_node():
    from core.graph_utils import apply_synthesis, combine, normalize_fragment

    fragment = {"nodes": [{"id": "local-1", "name": "Application Submission", "type": "activity"},
                          {"id": "local-2", "name": "Check fields", "type": "activity"}],
                "edges": [{"from": "local-1", "to": "local-2"}]}
    graph = combine([normalize_fragment(fragment, 1)])
    result = apply_synthesis(graph, {"merges": [{"ids": ["C1:local-1"], "type": "start"},
                                                {"ids": ["C1:local-2"], "name": "Check mandatory fields"}]})
    assert [n["type"] for n in result["nodes"]] == ["start", "activity"]
    assert result["nodes"][1]["name"] == "Check mandatory fields"
    assert len(result["nodes"]) == 2 and len(result["edges"]) == 1  # corrected, nothing merged or lost


def test_fresh_review_run_can_be_continued_without_fresh(tmp_path, monkeypatch, capsys):
    client = FakeClient()
    run_main(tmp_path, monkeypatch, client)                       # an older finished run
    run_main(tmp_path, monkeypatch, client, "--fresh", "--review")
    assert "Paused for review (1 of 2)" in capsys.readouterr().out
    run_main(tmp_path, monkeypatch, client, "--approve")          # no --fresh: must find the fresh run
    assert "Paused for review (2 of 2)" in capsys.readouterr().out
    run_main(tmp_path, monkeypatch, client, "--approve")
    assert "Done in" in capsys.readouterr().out
