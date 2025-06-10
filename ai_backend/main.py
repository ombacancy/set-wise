from utils.models import UserState
from langgraph_app import app

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        state = UserState(user_id="user_001", user_input=user_input)
        result = app.invoke(state)
        print("AI:", result.response)
