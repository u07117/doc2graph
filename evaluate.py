"""Evaluation loop: run every document in a dataset and score it against its reference graph.

Dataset layout (default eval/dataset/):
    loan_approval_policy.md          the document (any supported format)
    loan_approval_policy.gold.json   its reference graph (same JSON shape as the output)

Usage:
    python evaluate.py                         run + score the whole dataset
    python evaluate.py --save-baseline         ...and store the scores as the baseline
    python evaluate.py --score PRED.json GOLD.json [--document DOC]   score one existing output

Each run is compared with eval/baseline.json (if present). The command exits with
code 1 when an average metric drops by more than --tolerance, so it can gate
prompt/model changes (e.g. in CI).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from core.document_reader import read_document
from core.evaluation import score
from core.llm_client import create_client, default_model
from core.runner import process_document
from core.workflow import load_json

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DOCUMENT_TYPES = (".pdf", ".docx", ".md", ".markdown", ".txt")
# Average metrics shown in the scorecard and checked against the baseline.
KEY_METRICS = ["node_f1", "edge_f1", "decision_recall", "condition_accuracy",
               "actor_accuracy", "grounding_rate"]


def parse_args():
    parser = argparse.ArgumentParser(description="Score generated graphs against reference graphs.")
    parser.add_argument("dataset", nargs="?", default="eval/dataset")
    parser.add_argument("--score", nargs=2, metavar=("PRED", "GOLD"), help="Score one existing graph and exit.")
    parser.add_argument("--document", help="Source document for grounding when using --score.")
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--chunk-size", type=int, default=12000)
    parser.add_argument("--chunk-overlap", type=int, default=800)
    parser.add_argument("--merge-batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fresh", action="store_true", help="Recompute instead of reusing finished runs.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Name similarity needed to match a node.")
    parser.add_argument("--baseline", default="eval/baseline.json")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Allowed drop before failing.")
    parser.add_argument("--results-dir", default="eval/results")
    return parser.parse_args()


def find_cases(dataset):
    cases = []
    for gold in sorted(Path(dataset).glob("*.gold.json")):
        stem = gold.name[: -len(".gold.json")]
        docs = [p for p in gold.parent.glob(f"{stem}.*") if p.suffix.lower() in DOCUMENT_TYPES]
        if docs:
            cases.append((stem, docs[0], gold))
        else:
            print(f"warning: no document found for {gold.name}", file=sys.stderr)
    return cases


def fmt(value):
    if value is None:
        return "   -  "
    if isinstance(value, bool):
        return "  yes " if value else "  NO  "
    return f"{value:6.2f}"


def averages(rows):
    result = {}
    for key in KEY_METRICS:
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key] = round(sum(values) / len(values), 3) if values else None
    return result


def print_scorecard(rows):
    header = f"{'document':28} " + " ".join(f"{k[:10]:>10}" for k in KEY_METRICS) + "  class  issues"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['document'][:28]:28} " + " ".join(f"{fmt(r.get(k)):>10}" for k in KEY_METRICS)
              + f" {fmt(r.get('classification_correct'))} {r['structural_issues']:>6}")
    avg = averages(rows)
    print("-" * len(header))
    print(f"{'AVERAGE':28} " + " ".join(f"{fmt(avg[k]):>10}" for k in KEY_METRICS))
    return avg


def compare_with_baseline(avg, baseline_path, tolerance):
    path = Path(baseline_path)
    if not path.exists():
        print(f"\nNo baseline yet ({path}). Save one with --save-baseline.")
        return True
    baseline = json.loads(path.read_text())["averages"]
    print(f"\nChange vs baseline ({path}):")
    ok = True
    for key in KEY_METRICS:
        new, old = avg.get(key), baseline.get(key)
        if new is None or old is None:
            continue
        delta = new - old
        flag = "  REGRESSION" if delta < -tolerance else ""
        ok &= not flag
        print(f"  {key:20} {old:6.3f} -> {new:6.3f} ({delta:+.3f}){flag}")
    return ok


def main():
    args = parse_args()

    if args.score:
        pred, gold = load_json(args.score[0]), load_json(args.score[1])
        text = read_document(args.document) if args.document else None
        print(json.dumps(score(pred, gold, text, args.threshold), indent=2, ensure_ascii=False))
        return

    cases = find_cases(args.dataset)
    if not cases:
        raise SystemExit(f"No cases in {args.dataset} (need <name>.<ext> + <name>.gold.json).")
    try:
        client = create_client()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    rows, details = [], {}
    for stem, document, gold_path in cases:
        print(f"Running {document.name} ...", file=sys.stderr)
        result = process_document(
            document, client, model=args.model, output_dir=Path("eval/runs") / stem,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            merge_batch_size=args.merge_batch_size, concurrency=args.concurrency,
            fresh=args.fresh, log=lambda message: print(message, file=sys.stderr),
        )
        pred = load_json(result["graph_file"])
        metrics = score(pred, load_json(gold_path), read_document(document), args.threshold)
        rows.append({"document": stem, **{k: v for k, v in metrics.items()
                                          if k not in ("missed_nodes", "extra_nodes", "missed_edges")}})
        details[stem] = metrics

    print()
    avg = print_scorecard(rows)
    for stem, metrics in details.items():
        if metrics["missed_nodes"] or metrics["missed_edges"]:
            print(f"\n{stem}: missed {len(metrics['missed_nodes'])} node(s), {len(metrics['missed_edges'])} edge(s)")
            for name in metrics["missed_nodes"][:5]:
                print(f"  - missed node: {name}")
            for edge in metrics["missed_edges"][:5]:
                print(f"  - missed edge: {edge}")

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model,
              "settings": {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
                           "merge_batch_size": args.merge_batch_size, "threshold": args.threshold},
              "averages": avg, "documents": details}
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report: {out}")

    ok = compare_with_baseline(avg, args.baseline, args.tolerance)
    if args.save_baseline:
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.baseline).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved baseline: {args.baseline}")
    elif not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
