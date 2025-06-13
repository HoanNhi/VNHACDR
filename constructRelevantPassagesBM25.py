import bm25s
import os
import pytrec_eval
import json
from underthesea import word_tokenize

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
    return " ".join(tokenize_clean)

def judge(prev_data, data, turn, queryRaws, relevantPassages, relevantQueries, queryReform, scores):
    query_id = f"{data['session_id']}_{turn['turn_id']}"
    queryRaw = queryRaws[query_id]

    results = retriever.retrieve(tokenize_query(queryRaw), k=5, return_as = "documents")

    # Build qrel and run
    qrel = {query_id: {str(turn["pid"]): 1}}
    run = {query_id: {str(doc.metadata["pid"]): 1.0 / (i + 1) for i, doc in enumerate(results)}}

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
    scores = evaluator.evaluate(run)
    raw_score = scores[query_id]['recip_rank']
    relevantPassages[query_id] = []
    relevantQueries[query_id] = []
    # new_score = raw_score
    for prev_datum in prev_data:
        for prev_turn in prev_datum["turns"]:
            prev_qid = f"{prev_datum['session_id']}_{prev_turn['turn_id']}"
            if prev_datum["session_id"] == data["session_id"] and prev_turn["turn_id"] >= turn["turn_id"]:
                break
            new_query = f"{prev_turn['new_question']} "
            new_query_temp = new_query + queryRaw
            new_query_temp_with_passage = new_query + prev_turn['passage'] + " " + queryRaw
            new_query_results = vectorstore.similarity_search(new_query_temp, k=5)
            new_query_results_with_passage = vectorstore.similarity_search(new_query_temp_with_passage, k=5)
            run_new = {
                query_id: {str(doc.metadata["pid"]): 1.0 / (i + 1) for i, doc in enumerate(new_query_results)}
            }
            run_new_with_passage = {
                query_id: {str(doc.metadata["pid"]): 1.0 / (i + 1) for i, doc in enumerate(new_query_results_with_passage)}
            }

            # Evaluate with pytrec_eval
            evaluator_new = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
            scores_new = evaluator_new.evaluate(run_new)
            scores_new_with_passage = evaluator_new.evaluate(run_new_with_passage)
            raw_score_new = scores_new[query_id]['recip_rank']
            raw_score_new_with_passage = scores_new_with_passage[query_id]['recip_rank']

            if (raw_score_new > raw_score) or (raw_score_new_with_passage > raw_score):
                if raw_score_new > raw_score_new_with_passage:
                    # new_score = raw_score_new
                    best_query = new_query_temp
                    queryReform[query_id] = best_query
                    relevantQueries[query_id].append((prev_qid, prev_turn['new_question']))
                elif raw_score_new_with_passage >= raw_score_new:
                    # new_score = raw_score_new_with_passage
                    best_query = new_query_temp_with_passage
                    queryReform[query_id] = best_query
                    relevantQueries[query_id].append((prev_qid, prev_turn['new_question']))
                    relevantPassages[query_id].append((prev_turn["pid"], prev_turn["passage"]))

    if (len(relevantPassages[query_id]) == 0):
        relevantPassages[query_id] = [(turn["pid"], turn["passage"])]
        if (len(relevantQueries[query_id]) == 0):
            new_score = raw_score
            queryReform[query_id] = queryRaw

    # print(f"Original score is {raw_score}")
    # print(f"After updating, the new score is {new_score}. Current query is: {queryReform[query_id]}")
    print(f"Finish examining for turn {query_id}")
    # scores[query_id] = (raw_score, new_score)


def pseudoRelevantPassage(mappingFile = "question_passage_mapping.json"):
    dataset = loadDataset(mappingFile)
    relevantPassages = dict()
    relevantQueries = dict()
    queryReform = dict()
    queryRaws = dict()
    scores = dict()

    for i, data in enumerate(dataset):
        prev_data = [dataset[j] for j in range(i+1)]
        for turn in data["turns"]:
            key = str(data["session_id"]) + "_" + str(turn["turn_id"])
            if data["session_id"] == 0 and turn["turn_id"] == 0:
                relevantPassages[key] = [(turn["pid"], turn["passage"])]
                relevantQueries[key] = []
                queryReform[key] = turn["new_question"]
                queryRaws[key] =  turn["new_question"]
                scores[key] = (0, 0)
                continue
            else:
                queryRaws[key] =  turn["new_question"]
                judge(prev_data, data, turn, queryRaws, relevantPassages, relevantQueries, queryReform, scores)
    return relevantPassages, relevantQueries, queryReform, queryRaws, scores

def write_results_to_json(relevantPassages, relevantQueries, queryReform, queryRaws, scores, output_file="reformulated_queries.json"):
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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

def main():
    relevantPassages, relevantQueries, queryReform, queryRaws, scores = pseudoRelevantPassage("question_passage_mapping.json")
    write_results_to_json(relevantPassages, relevantQueries, queryReform, queryRaws, scores)
if __name__ == "__main__":
    main()




