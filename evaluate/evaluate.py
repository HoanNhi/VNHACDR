import json
import argparse
from pathlib import Path

import pytrec_eval
def load_json(path: Path):
    """Load and return the JSON list stored at *path*."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_qrels(gt_items):
    qrels = {}
    for obj in gt_items:
        qid = str(obj["id"])
        qrels[qid] = {str(p["pid"]): 1 for p in obj.get("pids", [])}
    return qrels


def build_run(pred_items):
    """
    Build a run dict for pytrec_eval:
        { query_id: { doc_id: score, ... }, ... }
    Scores are 1 / rank so higher ranks get higher scores.
    """
    run = {}
    for obj in pred_items:
        qid = str(obj["id"])
        run[qid] = {
            str(p["pid"]): 1.0 / p["rank"]        # simple descending score
            for p in obj.get("pids", [])
        }
    return run


def aggregate(metric_result):
    flat = {}
    for qres in metric_result.values():
        for m, v in qres.items():
            flat.setdefault(m, []).append(v)
    return {m: sum(lst) / len(lst) for m, lst in flat.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval with pytrec_eval")
    parser.add_argument("--qrels", type=Path, required=True, help="Ground‑truth JSON file")
    parser.add_argument("--run",   type=Path, required=True, help="Prediction JSON file")
    parser.add_argument("--result", type=Path, required=True, help="Result JSON file")
    parser.add_argument("--k",     type=int, default=100,     help="Cut‑off K")
    args = parser.parse_args()

    # 1. Load files
    gt_items   = load_json(args.qrels)
    pred_items = load_json(args.run)

    # 2. Build pytrec_eval structures
    qrels = build_qrels(gt_items)
    run   = build_run(pred_items)

    # 3. Define metrics
    k         = args.k
    metrics   = {
        "map",                       # Mean Average Precision (un‑cut)
        "recip_rank",                # MRR
        # f"P_{k}",            # Precision@K
        f"recall_{5}",               # Recall@K
        f"recall_{10}",
        f"recall_{15}",
        f"recall_{20}",
        f"recall_{30}",
        f"recall_{100}",
    }

    # 4. Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results   = evaluator.evaluate(run)

    # 5. Aggregate & report
    avg = aggregate(results)

    print("\n▸  Evaluation summary")
    print(json.dumps(avg, indent=2))

    save_file = Path(args.result)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with save_file.open("w", encoding="utf-8") as f:
        json.dump(avg, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
