def log_workout_node(state):
    # Logic to log to database or append to user profile
    workout_data = state.entities
    state.response = f"Got it! Logged your workout: {workout_data.get('exercise', [])} with {workout_data.get('reps', '?')} reps."
    return state