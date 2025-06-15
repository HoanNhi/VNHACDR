import json
from pathlib import Path

# --- Configuration ----------------------------------------------------------
INPUT_FILE  = Path("reformulated_queries.json")      # your source file
OUTPUT_FILE = Path("reformulated_queries_remove_duplicate.json")       # where to save the result
# ----------------------------------------------------------------------------

def deduplicate_passages(passages):
    """
    Return a list of unique passages *after* removing the first item.
    Uniqueness is defined by the (pid, passage) pair so we don’t lose
    information if two different ids point to identical text or vice‑versa.
    """
    seen = set()
    unique = []
    for p in passages[1:]:            # skip the very first passage
        key = (p.get("pid"), p.get("passage"))
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique

def main() -> None:
    # 1. Load input
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        items = json.load(f)

    # 2. Transform each item
    for obj in items:
        passages = obj.get("relevant_passages")
        if not isinstance(passages, list):
            continue                         # nothing to do if it's missing/invalid

        if len(passages) == 1:
            # Remove the entire key
            obj.pop("relevant_passages", None)
        elif len(passages) > 1:
            # Remove first passage & deduplicate the rest
            obj["relevant_passages"] = deduplicate_passages(passages)

    # 3. Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(items)} items ➜ {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
