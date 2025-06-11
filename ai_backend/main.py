from schema import IntentPayload, WorkoutLog, UserStatus, UserGoal
from langgraph_flow import build_graph
import uuid


# Simulate user input collection (replace with actual input in production)
def get_user_input():
    return {
        "user_id": str(uuid.uuid4()),
        "raw_input": input("\n💬 You: "),
        "status": {
            "soreness": ["legs", "shoulders"],
            "energy": 6,
            "mood": "motivated"
        },
        "goal": {
            "target": "build muscle",
            "focus_areas": ["arms", "chest"]
        }
    }


# Convert raw input into structured IntentPayload
def create_payload(data: dict) -> IntentPayload:
    return IntentPayload(
        user_id=data["user_id"],
        raw_input=data["raw_input"],
        status=UserStatus(**data.get("status", {})),
        goal=UserGoal(**data.get("goal", {}))
    )


def main():
    print("🏋️ Welcome to Gym AI Buddy 💪")
    flow = build_graph()

    user_data = get_user_input()
    payload = create_payload(user_data)

    # Run LangGraph flow
    result = flow.invoke(payload)

    print("\n✅ Flow finished.")


if __name__ == "__main__":
    main()
