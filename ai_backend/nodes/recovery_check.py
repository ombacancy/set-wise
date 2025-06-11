from schema import IntentPayload
from llm import simple_prompt

class RecoveryCheckNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        soreness = ", ".join(state.status.soreness) if state.status and state.status.soreness else "unknown"
        energy = state.status.energy if state.status and state.status.energy else "unknown"

        prompt = (
            f"The user has soreness in: {soreness} and energy level is {energy}/10.\n"
            "Should they work out today? Provide reasoning and alternative advice if needed."
        )

        advice = simple_prompt(prompt, system="You're a physical therapist AI.")
        print("\n🛌 Recovery Advice:\n", advice)
        return state
