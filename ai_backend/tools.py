from langchain_core.tools import tool


@tool
def track_goal(user_data: dict) -> dict:
    """
    Tracks user's goal progress based on recent workout behavior.

    Args:
        user_data (dict): {
            'goal': str,  # e.g., 'muscle gain', 'weight loss'
            'weekly_workouts': int,
            'weekly_minutes': int
        }

    Returns:
        dict: Analysis of goal adherence
    """
    goal = user_data["goal"]
    workouts = user_data["weekly_workouts"]
    minutes = user_data["weekly_minutes"]

    if goal == "muscle gain":
        status = "on track" if workouts >= 4 and minutes >= 120 else "behind"
    elif goal == "weight loss":
        status = "on track" if minutes >= 150 else "behind"
    else:
        status = "tracking not defined"

    return {
        "goal": goal,
        "status": status,
        "message": "Keep going!" if status == "on track" else "Let's step it up next week!"
    }


@tool
def explain_workout(data: dict) -> dict:
    """
    Explains a workout by name with image, target muscle and equipment.

    Args:
        data (dict): { "workout_name": str }

    Returns:
        dict: Details of the workout
    """
    workout_name = data["workout_name"].lower()

    with open("data/workouts.json") as f:
        workouts = json.load(f)

    for workout in workouts:
        if workout["name"].lower() == workout_name:
            return {
                "name": workout["name"],
                "target_muscle": workout["target_muscle"],
                "equipment": workout["equipment"],
                "gif_url": workout["gif_url"],
                "explanation": workout["explanation"]
            }

    return {"error": f"Workout '{data['workout_name']}' not found."}


@tool
def analyze_fitness(workout_data: dict) -> dict:
    """
    Analyzes workout and provides personalized recommendations.

    Args:
        workout_data (dict): Dictionary containing:
            - workout_type: str (e.g., 'cardio', 'strength')
            - duration_minutes: int
            - intensity_level: str ('low', 'medium', 'high')
            - frequency_per_week: int

    Returns:
        dict: Analysis results and recommendations
    """

    def calculate_recommendations(data):
        # Calculate optimal workout parameters
        base_duration = data['duration_minutes']
        current_frequency = data['frequency_per_week']
        intensity = data['intensity_level']

        # Calculate recommended progression
        if intensity == 'low':
            recommended_duration = min(base_duration + 10, 60)
            recommended_frequency = min(current_frequency + 1, 5)
        elif intensity == 'medium':
            recommended_duration = min(base_duration + 5, 45)
            recommended_frequency = min(current_frequency + 1, 4)
        else:  # high intensity
            recommended_duration = base_duration
            recommended_frequency = min(current_frequency, 3)

        return {
            "current_analysis": {
                "workout_level": intensity,
                "weekly_minutes": base_duration * current_frequency
            },
            "recommendations": {
                "target_duration": recommended_duration,
                "target_frequency": recommended_frequency,
                "next_intensity": "medium" if intensity == "low" else "high",
                "rest_days": max(7 - recommended_frequency, 2)
            }
        }

    return calculate_recommendations(workout_data)
