import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# ---------- Config ----------
INPUT_PATH  = Path("reformulated_queries.json")
OUTPUT_PATH = Path("results/retrieval_result_dense_100.json")
PERSIST_DIR = Path("./chroma_store")
COLLECTION  = "viWiki"
TOP_K       = 100
MODEL_NAME  = "AITeamVN/Vietnamese_Embedding"
# ----------------------------

# 1. Load the jsonl file
with INPUT_PATH.open("r", encoding="utf-8") as f:
    records = json.load(f)

# 2. Init embedding model & Chroma vector store
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
vectorstore = Chroma(collection_name=COLLECTION,
                     embedding_function=embedding_model,
                     persist_directory=str(PERSIST_DIR))

# 3. Process and collect results
all_results = []

for record in records:
    sample_id = record.get("id", "unknown_id")

    # Compose the new query
    query_parts = []

    # (1) Add relevant_query elements
    if "relevant_query" in record and isinstance(record["relevant_query"], list):
        for relevant_query in record["relevant_query"]:
            prev_query = relevant_query["query"]
            if prev_query:
                query_parts.append(prev_query)

    # (2) Add relevant_passages if it exists
    if "relevant_passages" in record and isinstance(record["relevant_passages"], list):
        for passage in record["relevant_passages"]:
            passage_text = passage.get("passage")
            if passage_text:
                query_parts.append(passage_text)

    # (3) Add the original_question
    original_question = record.get("original_question", "")
    query_parts.append(original_question.strip())

    # Final query string
    print(query_parts)
    full_query = " ".join(query_parts).strip()

    if not full_query:
        print(f"[{sample_id}] Empty query. Skipping.")
        all_results.append({"id": sample_id, "pids": []})
        continue

    # Embed and retrieve
    query_embedding = embedding_model.embed_query(full_query)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=TOP_K)

    # Collect top-k pids with rank
    pids = [
        {
            "pid": doc.metadata.get("pid"),
            "rank": i + 1
        }
        for i, doc in enumerate(results)
    ]

    all_results.append({
        "id": sample_id,
        "pids": pids
    })

# 4. Save to output file
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_results)} entries to {OUTPUT_PATH}")
