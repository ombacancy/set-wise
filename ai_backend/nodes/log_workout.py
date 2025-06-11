# nodes/log_workout.py

from schema import IntentPayload
from vdb import store_log
from datetime import datetime


class LogWorkoutNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        if state.workout:
            log_text = (
                f"{state.workout.exercise}, "
                f"{state.workout.sets or '?'} sets, "
                f"{state.workout.reps or '?'} reps, "
                f"{state.workout.weight or '?'} kg, "
                f"notes: {state.workout.notes or 'None'}"
            )
            metadata = {
                "user_id": state.user_id,
                "exercise": state.workout.exercise,
                "date": state.workout.date.isoformat(),
                "reps": state.workout.reps,
                "sets": state.workout.sets,
                "weight": state.workout.weight,
                "notes": state.workout.notes,
            }
            store_log(state.user_id, log_text, metadata)
        return state
