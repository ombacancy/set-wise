from schema import IntentPayload, UserGoal

class GoalUpdateNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        if state.goal:
            print(f"\n✅ Updated goal to: {state.goal.target}, Focus areas: {state.goal.focus_areas}")
        else:
            print("\n⚠️ No valid goal provided to update.")
        return state
