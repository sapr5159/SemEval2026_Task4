"""
Track A baseline system.

We use a naive prompt for chatGPT.
"""

import os
import random
from enum import Enum

from openai import OpenAI
from groq import Groq
import pandas as pd
from pydantic import BaseModel


class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum


def predict(row):
    """
    Uses the OpenAI API to determine which of two stories (A or B) is more narratively similar to an anchor story.

    Returns:
        bool: True if story A is predicted to be more similar to the anchor than story B; False otherwise.
    """
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    completion = client.chat.completions.parse(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert on stories and narratives. Tell us which of two stories is narratively similar to the anchor story.",
            },
            {
                "role": "user",
                "content": f"Anchor story: {anchor}\n\nStory A: {text_a}\n\nStory B: {text_b}",
            },
        ],
        response_format=SimilarityPrediction,
    )
    return completion.choices[0].message.parsed == ResponseEnum.A


key = os.environ.get("GROQ_API_KEY")
if not key:
    raise RuntimeError("GROQ_API_KEY not set. Use `export GROQ_API_KEY=...` (Windows: setx GROQ_API_KEY ...)")

baseline = "groq"
df = pd.read_json("data/dev_track_a.jsonl", lines=True)
if baseline == "groq":
    client = Groq(api_key=key)
    df["predicted_text_a_is_closer"] = df.apply(predict, axis=1)
elif baseline == "random":
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )
print(df["predicted_text_a_is_closer"])
accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
print(f"Accuracy: {accuracy:.3f}")


df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

open("output/track_a.jsonl", "w").write(df.to_json(orient='records', lines=True))
