# nodes/plan_suggestion.py

from schema import IntentPayload
from vdb import query_logs
from llm import simple_prompt

class PlanSuggestionNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        past_logs = query_logs(state.user_id, "recent workouts", k=5)
        history = "\n".join([doc for doc in past_logs["documents"][0]])

        energy = state.status.energy if state.status and state.status.energy else "unknown"
        soreness = ", ".join(state.status.soreness) if state.status and state.status.soreness else "unknown"
        goal = state.goal.target if state.goal else "general fitness"

        prompt = (
            f"My energy level today is {energy} out of 10. "
            f"I'm sore in: {soreness}. "
            f"My goal is: {goal}.\n"
            f"My recent workouts:\n{history}\n"
            "Suggest a personalized workout plan for today."
        )

        plan = simple_prompt(prompt, system="You are a certified fitness trainer AI.")
        print("\n📌 Suggested Plan:\n", plan)
        return state
