import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

def mapQuestionToPassage(question_path, passage_path):
    question_data = pd.read_csv(question_path)
    passage_data = pd.read_csv(passage_path)

    passages_df = passage_data.fillna('')

    # Create a dictionary to lookup passage by pid
    passage_lookup = {
        row['pid']: f"{row['section_title']} {row['title']} {row['section_text']}"
        for _, row in passages_df.iterrows()
    }

    # Group by session_id
    grouped = {}

    for _, row in tqdm(question_data.iterrows()):
        session_id = row['session_id']
        turn_id = row['turn_id']
        new_question = row['new_question']
        pid = row['pid']
        passage = passage_lookup.get(pid, '')

        turn_data = {
            "turn_id": int(turn_id),
            "new_question": new_question if pd.notna(new_question) else "",
            "pid": pid,
            "passage": passage
        }

        if session_id not in grouped:
            grouped[session_id] = {
                "session_id": session_id,
                "turns": []
            }

        grouped[session_id]["turns"].append(turn_data)

    # Convert to list of session-wise JSON objects
    json_output = list(grouped.values())

    # Save to file
    with open("question_passage_mapping.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

def main():
    mapQuestionToPassage("datasets/conversation_topic_shifted.test.csv", "datasets/viwiki-passages.csv")

if __name__ == "__main__":
    main()