from schema import IntentPayload
from llm import simple_prompt


class FallbackNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        prompt = f"User said: {state.raw_input}\nThis wasn't clearly understood. Ask a clarifying question."
        clarification = simple_prompt(prompt, system_prompt="You're a friendly fitness assistant.")
        print("\n❓ Clarification:\n", clarification)
        return state
