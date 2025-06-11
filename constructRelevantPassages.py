from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import numpy as np
import pytrec_eval
import os
import json

embedding_model = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")
vectorstore = Chroma(
    collection_name="viWiki",
    embedding_function=embedding_model,
    persist_directory="./chroma_store"
)

def loadDataset(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def judge(data, turn, queryRaws, relevantPassages, queryReform):
    queryRaw = turn["new_question"]
    query_id = f"{data['session_id']}_{turn['turn_id']}"  # Ensure string keys for pytrec_eval

    # Embed and search
    queryRawEmbedded = embedding_model.embed_documents(queryRaw)
    results = vectorstore.similarity_search_by_vector(queryRawEmbedded.tolist(), k=3)

    # Build qrel: ground truth pids with relevance 1
    qrel = {
        query_id: {pid: 1 for pid in relevantPassages}
    }

    # Build run: retrieved pids with dummy scores (e.g., inverse of rank)
    run = {
        query_id: {doc['pid']: 1.0 / (i + 1) for i, doc in enumerate(results)}
    }

    # Evaluate with pytrec_eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
    scores = evaluator.evaluate(run)
    raw_score = scores[query_id]['recip_rank']

    bestQuery = None
    for key in relevantPassages.keys():
        new_query = f"{queryRaw} {queryRaws[key]}"
        for passage in relevantPassages[key]:
            new_query += " " + passage
        new_queryEmbedded = embedding_model.embed_documents(queryRaw)
        new_query_results = vectorstore.similarity_search_by_vector(new_queryEmbedded.tolist(), k=3)

        # Build qrel: ground truth pids with relevance 1
        qrel_new = {
            query_id: {pid: 1 for pid in relevantPassages}
        }

        # Build run: retrieved pids with dummy scores (e.g., inverse of rank)
        run_new = {
            query_id: {doc['pid']: 1.0 / (i + 1) for i, doc in enumerate(new_query_results)}
        }

        # Evaluate with pytrec_eval
        evaluator_new = pytrec_eval.RelevanceEvaluator(qrel_new, {'recip_rank'})
        scores_new = evaluator_new.evaluate(run_new)
        raw_score_new = scores_new[query_id]['recip_rank']

        if (raw_score_new > raw_score):
            bestQuery = new_query
            queryReform[query_id] = bestQuery
            relevantPassages[query_id] = [turn["passage"]]
            relevantPassages[query_id].addAll(relevantPassages[key])



def pseudoRelevantPassage():
    mappingFile = "question_passage_mapping.json"
    dataset = loadDataset(mappingFile)
    relevantPassages = dict()
    queryReform = dict()
    queryRaws = dict()

    for data in dataset:
        for turn in data["turns"]:
            key = str(data["session_id"]) + "_" + str(turn["turn_id"])
            if data["session_id"] == 0 and turn["turn_id"] == 0:
                relevantPassages[key] = [turn["passage"]]
                queryReform[key] = turn["new_question"]
                queryRaws[key] =  turn["new_question"]
                continue
            else:
                queryRaws[key] =  turn["new_question"]
                judge(data, turn, queryRaws, relevantPassages, queryReform)




