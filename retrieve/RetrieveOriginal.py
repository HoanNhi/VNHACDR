#!/usr/bin/env python3
"""
Create two retrieval‑prediction files:

1. new_question_predictions.json  – uses the `new_question` field
2. rewrite_predictions.json       – uses the `rewrite`       field

Each prediction file is a list of:
{
  "id":   "<session_id>_<turn_id>",
  "pids": [
      { "pid": <int>, "rank": 1 },
      ...
      { "pid": <int>, "rank": 100 }
  ]
}

Requirements:
  pip install pandas langchain_huggingface langchain-community chromadb
"""

from pathlib import Path
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── CONFIG ──────────────────────────────────────────────────────────────────
CSV_PATH     = Path("datasets/conversation_topic_shifted.test.csv")            # <- your CSV file
PERSIST_DIR  = Path("./chroma_store")                    # <- Chroma directory
COLLECTION   = "viWiki"                                  # <- Chroma collection name
MODEL_NAME   = "AITeamVN/Vietnamese_Embedding"           # <- HF embedding model
TOP_K        = 100                                       # <- number of docs
OUT_NEW_Q    = Path("results/question_predictions.json")
OUT_REWRITE  = Path("results/rewrite_predictions.json")
# ────────────────────────────────────────────────────────────────────────────


def load_vectorstore():
    """Initialise embeddings + Chroma vector store once."""
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return embeddings, store


def search_and_format(df: pd.DataFrame,
                      query_col: str,
                      embeddings,
                      store,
                      top_k: int) -> list[dict]:
    """Embed each query (from *query_col*) and retrieve top‑k pids."""
    results = []

    for row in df.itertuples(index=False):
        qid        = f"{row.session_id}_{row.turn_id}"
        query_text = getattr(row, query_col, "") or ""

        if not query_text.strip():
            # Empty query ⇒ still write an entry (with no pids) for completeness
            results.append({"id": qid, "pids": []})
            continue

        emb    = embeddings.embed_query(query_text)
        docs   = store.similarity_search_by_vector(emb, k=top_k)

        pids = [
            {"pid": doc.metadata.get("pid"), "rank": rank + 1}
            for rank, doc in enumerate(docs)
        ]

        results.append({"id": qid, "pids": pids})

    return results


def main() -> None:
    # 1. Load CSV
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    # 2. Init vector store once
    embeddings, store = load_vectorstore()

    # 3. Build the two prediction lists
    preds_new_q = search_and_format(df,
                                    query_col="new_question",
                                    embeddings=embeddings,
                                    store=store,
                                    top_k=TOP_K)

    preds_rew   = search_and_format(df,
                                    query_col="rewrite",
                                    embeddings=embeddings,
                                    store=store,
                                    top_k=TOP_K)

    # 4. Write to JSON
    for path, data in ((OUT_NEW_Q, preds_new_q), (OUT_REWRITE, preds_rew)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} predictions → {path}")


if __name__ == "__main__":
    main()
