def suggest_plan_node(state):
    entities = state.entities
    # Very basic suggestion logic for now
    sore = entities.get("soreness", [])
    suggestions = "Today is a good day for legs and core since your arms are sore." if "biceps" in sore else "Let's hit arms and shoulders today!"
    state.response = suggestions
    return state
