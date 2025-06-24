import os
import json
import re
import string
import random
import pandas as pd
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    classification_report,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Union, List, Optional
import glob

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")


class Answer(BaseModel):
    """Question answer"""

    answer: str = Field(
        description="only one word from the possible answers containing your answer."
    )


disrpt_prompt_template = PromptTemplate.from_template(
    "Определите связь между двумя предложениями. Возможные следующие варианты ответа:\n {options}.\nПредложение 1: {start}\nПредложение 2: {end}\nДайте только один ответ из предложенных. Используйте JSON для вывода, состоящий из одного поля: `answer`."
)


def score_disrpt(model_name, temperature, test_set: pd.DataFrame, filename: str):
    """Run scoring on the disrpt.json dataset."""
    # Initialize model
    model = ChatOpenAI(
        model=model_name, api_key=API_KEY, base_url=BASE_URL, temperature=temperature
    )
    structured_llm = model.with_structured_output(Answer)

    # Generate model answers
    results = []
    print("[INFO] Starting disrpt scoring...")
    # Iterate through the test set
    for _, row in tqdm(test_set.iterrows(), total=len(test_set), desc="Scoring disrpt"):
        try:
            options = row["choices"]
            options.append(row["label"])
            res = structured_llm.invoke(
                disrpt_prompt_template.invoke(
                    {
                        "options": options,
                        "start": row["start"],
                        "end": row["end"],
                    }
                )
            )
            results.append((row, res, row["label"]))
        except Exception as e:
            print(e)

    # Calculate metrics
    y_true = [i[2].strip().lower() for i in results]
    y_pred = [i[1].answer.strip().lower() for i in results]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
    }

    # Save answers to CSV
    answers_df = pd.DataFrame(
        {
            "task_id": [i[0].get("id", idx) for idx, i in enumerate(results)],
            "answer": [
                i[1]["answer"] if isinstance(i[1], dict) else i[1].answer
                for i in results
            ],
            "correct_answer": [i[2] for i in results],
        }
    )
    answers_df.to_csv(filename, index=False)

    print("[INFO] Disrpt scoring complete. Results saved to:", filename)
    return metrics, filename
