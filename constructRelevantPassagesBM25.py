import time

import bm25s
import os
import pytrec_eval
import json
from underthesea import word_tokenize
import multiprocessing as mp


path = os.path.join("datasets", "bm25s_indices")
retriever = bm25s.BM25.load(path, load_corpus=True)

stop_words = set()
with open("datasets/vietnamese-stopwords.txt", "r") as f:
    for line in f.readlines():
        stop_words.add(line.strip())

def loadDataset(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def tokenize_query(query):
    query = query.strip().lower()
    query = query.replace("'''", "\"")
    query = query.replace("''", "\"")
    tokenize = word_tokenize(query)
    tokenize_clean = [token for token in tokenize if token not in stop_words]
    # print(len(tokenize_clean))
    return tokenize_clean

def _build_prev_items(args):
    """
    Build one triple for a previous turn.
    Returns None when the turn should be skipped.
    """
    (prev_session_id,
     prev_turn,
     cur_session_id,
     cur_turn_id,
     query_raw) = args

    # Skip turns that are in the *same* session *after* the current turn
    if prev_session_id == cur_session_id and prev_turn["turn_id"] >= cur_turn_id:
        return None

    prev_qid = f"{prev_session_id}_{prev_turn['turn_id']}"
    passage_entry = {
        "qid": prev_qid,
        "question": prev_turn["new_question"],
        "pid": prev_turn["pid"],
        "passage": prev_turn["passage"],
    }
    passage_query  = tokenize_query(
        f"{prev_turn['new_question']} {prev_turn['passage']} {query_raw}"
    )
    question_query = tokenize_query(
        f"{prev_turn['new_question']} {query_raw}"
    )

    # The main process will unpack this tuple
    return passage_entry, passage_query, question_query


# ---------------- the parallelised function ----------------
def prepare_question(prev_data, data, turn, queryRaw):
    query_id = f"{data['session_id']}_{turn['turn_id']}"

    # Initialise result structures
    previous_question_passage        = {query_id: []}
    previous_question_query          = {query_id: []}
    previous_question_passage_query  = {query_id: []}

    # 1)  Build a flat task list (very cheap compared to tokenisation)
    tasks = [
        (
            prev_datum["session_id"],
            prev_turn,
            data["session_id"],
            turn["turn_id"],
            queryRaw,
        )
        for prev_datum in prev_data
        for prev_turn in prev_datum["turns"]
    ]

    n_cpus = 20
    with mp.Pool(n_cpus) as pool:
        for result in pool.imap_unordered(_build_prev_items, tasks, chunksize=64):
            if result is None:        # turn was skipped
                continue
            passage_entry, passage_query, question_query = result
            previous_question_passage[query_id].append(passage_entry)
            previous_question_passage_query[query_id].append(passage_query)
            previous_question_query[query_id].append(question_query)

    return previous_question_passage, previous_question_query, previous_question_passage_query


def _score_candidate(args):
    """
    Evaluate one candidate turn in its own process.
    Returns None if the candidate is not better than raw_score,
    otherwise returns (relevant_query_tuple, relevant_passage_tuple_or_None).
    """
    (i,
     result_passage,          # results_new_question_passage[i]
     result_query,            # results_new_question[i]
     prev_meta,               # previous_question_passage[query_id][i]
     query_id,
     qrel,
     raw_score) = args

    # Build evaluator locally (cheap)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})

    run_passage = {query_id: {str(doc["id"]): 1.0 / (j + 1)
                              for j, doc in enumerate(result_passage)}}
    run_query   = {query_id: {str(doc["id"]): 1.0 / (j + 1)
                              for j, doc in enumerate(result_query)}}

    new_score_query        = evaluator.evaluate(run_query)[query_id]['recip_rank']
    new_score_query_passage = evaluator.evaluate(run_passage)[query_id]['recip_rank']

    # Skip if neither beats the raw score
    if (new_score_query  <= raw_score and
        new_score_query_passage <= raw_score):
        return None

    query_tuple = (prev_meta["qid"], prev_meta["question"])
    passage_tuple = None
    if new_score_query_passage >= new_score_query:        # passage wins or ties
        passage_tuple = (prev_meta["pid"], prev_meta["passage"])

    return (query_tuple, passage_tuple)

def judge(query_id,
          truePassages,
          previous_question_passage,
          previous_question_query,
          previous_question_passage_query,
          queryRaw,
          relevantPassages,
          relevantQueries):

    try:
        results_raw = retriever.retrieve([tokenize_query(queryRaw)], k=5, return_as = "documents")
    except Exception as e:
        print(e)
        print(queryRaw)
        print(tokenize_query(queryRaw))
        return

    # Build qrel and run
    qrel = {query_id: {str(truePassages[query_id]): 1}}
    run_raw = {query_id: {str(doc["id"]): 1.0 / (i + 1) for i, doc in enumerate(results_raw[0])}}
    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
    scores = evaluator.evaluate(run_raw)
    raw_score = scores[query_id]['recip_rank']

    results_new_question_passage = retriever.retrieve(previous_question_passage_query[query_id], k=5, n_threads = 20, show_progress = True, return_as = "documents")
    results_new_question = retriever.retrieve(previous_question_query[query_id], k=5, n_threads = 20, show_progress = True, return_as = "documents")
    relevantPassages[query_id] = set()
    relevantQueries[query_id] = set()

    payloads = [(i,
                 results_new_question_passage[i],
                 results_new_question[i],
                 previous_question_passage[query_id][i],
                 query_id,
                 qrel,
                 raw_score)
                for i in range(len(results_new_question_passage))]

    # 2. parallel map
    n_cpu = 20
    with mp.Pool(n_cpu) as pool:
        for res in pool.imap_unordered(_score_candidate, payloads, chunksize=16):
            if res is None:
                continue
            query_tuple, passage_tuple = res
            relevantQueries[query_id].add(query_tuple)
            if passage_tuple:  # only when passage beat/tied
                relevantPassages[query_id].add(passage_tuple)

    print(f"Finish examining for turn {query_id}")

    # for i in range(len(results_new_question_passage)):
    #     result_new_question_passage = results_new_question_passage[i]
    #     result_new_question = results_new_question[i]
    #     run_new_question_passage = {query_id: {str(doc["id"]): 1.0 / (j+1) for j, doc in enumerate(result_new_question_passage)}}
    #     run_new_question = {query_id: {str(doc["id"]) : 1.0 / (j+1) for j, doc in enumerate(result_new_question)}}
    #     scores_new_question_passage = evaluator.evaluate(run_new_question_passage)
    #     scores_new_question = evaluator.evaluate(run_new_question)
    #     new_score_question = scores_new_question[query_id]['recip_rank']
    #     new_score_question_passage = scores_new_question_passage[query_id]['recip_rank']
    #
    #     if (new_score_question <=raw_score) and (new_score_question_passage <=raw_score):
    #         continue
    #
    #     prev_turn_id = previous_question_passage[query_id][i]["qid"]
    #     prev_question = previous_question_passage[query_id][i]["question"]
    #     relevantQueries[query_id].add((prev_turn_id, prev_question))
    #     if new_score_question_passage >= new_score_question:
    #         prev_passage = previous_question_passage[query_id][i]["passage"]
    #         prev_pid = previous_question_passage[query_id][i]["pid"]
    #         relevantPassages[query_id].add((prev_pid, prev_passage))
    #
    # print(f"Finish examining for turn {query_id}")


def pseudoRelevantPassage(mappingFile = "question_passage_mapping.json"):
    dataset = loadDataset(mappingFile)
    previous_question_passages = []
    previous_question_queries = []
    previous_question_passage_queries = []
    relevantPassages = dict()
    relevantQueries = dict()
    truePassages = dict()
    queryRaws = dict()

    for i, data in enumerate(dataset):
        prev_data = [dataset[j] for j in range(i+1)]
        for turn in data["turns"]:
            key = str(data["session_id"]) + "_" + str(turn["turn_id"])
            if data["session_id"] == 0 and turn["turn_id"] == 0:
                relevantPassages[key] = []
                relevantQueries[key] = []
                queryRaws[key] =  turn["new_question"]
                truePassages[key] = turn["pid"]
                continue
            else:
                queryRaws[key] = turn["new_question"]
                truePassages[key] = turn["pid"]
                previous_question_passage, previous_question_query, previous_question_passage_query = prepare_question(prev_data, data, turn, turn["new_question"])
                previous_question_passages.append(previous_question_passage)
                previous_question_queries.append(previous_question_query)
                previous_question_passage_queries.append(previous_question_passage_query)
                judge(key,
                      truePassages,
                      previous_question_passage,
                      previous_question_query,
                      previous_question_passage_query,
                      turn["new_question"],
                      relevantPassages,
                      relevantQueries)



    return relevantPassages, relevantQueries, queryRaws

def write_results_to_json(
        relevantPassages,
        relevantQueries,
        queryRaws,
        time_taken,
        output_file_name="reformulated_queries.json"):
    output_data = []

    for query_id in relevantPassages:
        item = {
            "id": query_id,
            "original_question": queryRaws.get(query_id, ""),
            # "query": queryReform.get(query_id, ""),
            "relevant_passages": [
                {
                    "pid": pid,
                    "passage": passage
                }
                for pid, passage in relevantPassages[query_id]
            ],
            "relevant_query":[
                {
                    "qid": qid,
                    "query": query
                }
                for qid, query in relevantQueries[query_id]
            ],
            # "old_score": scores[query_id](0),
            # "new_score": scores[query_id](1)
        }
        output_data.append(item)

    output_file = f"{output_file_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    time_file_name = f"{output_file_name}_time"
    with open(time_file_name, "w", encoding="utf-8") as f:
        f.write(str(time_taken))

def main():
    start_time = time.time()
    relevantPassages, relevantQueries, queryRaws = pseudoRelevantPassage("question_passage_mapping.json")
    time_taken = time.time() - start_time
    write_results_to_json(relevantPassages, relevantQueries, queryRaws, time_taken, output_file_name="test_reformulated_bm25")
if __name__ == "__main__":
    main()




