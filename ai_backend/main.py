from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
import json

app = FastAPI()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = 'https://api.groq.ai/v1/llm'


class ParseRequest(BaseModel):
    text: str


@app.post("/parse-workout/")
def parse_workout(req: ParseRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    prompt = f"""
    You are an expert fitness coach. Parse the following workout log into JSON with fields:
    - exercises: list of {{"name", "sets", "reps", "weight", "notes"}}
    - muscles_sore: list of muscles sore or fatigued
    - energy_level: estimated energy left (high/medium/low)
    Workout log: \"\"\"{req.text}\"\"\"
    Return only JSON.
    """

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.5,
        "top_p": 1,
        "n": 1,
        "stop": None,
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    parsed_text = data['choices'][0]['text']

    try:
        return json.loads(parsed_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response", "raw": parsed_text}
