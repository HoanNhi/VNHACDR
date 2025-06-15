import json
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE  = Path("question_passage_mapping.json")      # <— your source file
OUTPUT_FILE = Path("results/ground_truth.json")          # <— destination
# ────────────────────────────────────────────────────────────────────────────

def extract_pids(turn: dict) -> list[int]:
    """
    Returns a list of pids for a turn.
    • If the turn already has a list called `pids`, use it.
    • Otherwise fall back to the single `pid` field (wrap in a list).
    """
    if isinstance(turn.get("pids"), list):
        return turn["pids"]
    single_pid = turn.get("pid")
    return [single_pid] if single_pid is not None else []

def main() -> None:
    # 1. Load the conversation data
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        sessions = json.load(f)

    # 2. Flatten sessions → turns, assign ids, rank pids
    output = []
    for session in sessions:
        sid = session.get("session_id")
        for turn in session.get("turns", []):
            tid  = turn.get("turn_id")
            pids = extract_pids(turn)

            output.append({
                "id": f"{sid}_{tid}",
                "pids": [
                    {"pid": pid, "rank": rank + 1}
                    for rank, pid in enumerate(pids)
                ]
            })

    # 3. Write the transformed list back to disk
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(output)} turns → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
