# nodes/progress.py

from schema import IntentPayload
from vdb import query_logs
from llm import simple_prompt

class ProgressNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        logs = query_logs(state.user_id, "progress summary", k=10)
        history = "\n".join(logs["documents"][0])

        prompt = (
            f"Based on this user's workout logs:\n{history}\n"
            "Summarize their recent fitness progress, improvements, and suggest areas to focus on."
        )

        summary = simple_prompt(prompt, system="You're a fitness data analyst AI.")
        print("\n📊 Progress Summary:\n", summary)
        return state
