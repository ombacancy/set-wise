from schema import IntentPayload, WorkoutLog, UserGoal, UserStatus
from llm import simple_prompt
import json

INTENT_SYSTEM_PROMPT = """
You're a fitness assistant AI. Extract the user's intent and relevant data.
Recognized intents: [log_workout, update_goal, get_plan, check_progress, motivate, recovery_check]

Return a JSON object with:
- intent: string
- workout (if mentioned): {exercise, reps, sets, weight, notes}
- goal (if mentioned): {target, focus_areas}
- status (if mentioned): {soreness, energy, mood}
"""

class ParseIntentNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        prompt = f"User said: {state.raw_input}"
        output = simple_prompt(prompt, INTENT_SYSTEM_PROMPT)
        try:
            parsed = json.loads(output)
            state.intent = parsed.get("intent", "unknown")
            if "workout" in parsed:
                state.workout = WorkoutLog(**parsed["workout"])
            if "goal" in parsed:
                state.goal = UserGoal(**parsed["goal"])
            if "status" in parsed:
                state.status = UserStatus(**parsed["status"])
        except Exception:
            state.intent = "fallback"
        return state
