import os
import requests

LLM_SERVICE_URL = os.getenv('LLM_SERVICE_URL', 'http://localhost:8001/parse-workout/')


class FastAPILLMClient:
    def __init__(self, base_url=LLM_SERVICE_URL):
        self.url = base_url

    def parse_workout(self, text):
        try:
            response = requests.post(self.url, json={"text": text})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"LLM service call failed: {str(e)}"}
