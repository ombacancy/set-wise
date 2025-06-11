# nodes/motivation.py

from schema import IntentPayload
from llm import simple_prompt


class MotivationNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        prompt = f"User: {state.raw_input}\nGive a short motivational message to a gym-goer based on this."
        message = simple_prompt(prompt, system="You're a motivational fitness coach.")
        print("\n🔥 Motivation:\n", message)
        return state
