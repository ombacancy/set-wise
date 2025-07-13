# ai/main.py

import os
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from datetime import datetime

# LangChain imports
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain.globals import set_verbose

# LangGraph imports
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

# Vector DB - already using the correct imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM
from llm import llm

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Enable verbose mode for debugging
set_verbose(True)


# Vector DB setup
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create or load vector stores for different data types
health_store = Chroma(
    collection_name="health_records",
    embedding_function=embeddings,
    persist_directory="./data/chroma/health"
)

goals_store = Chroma(
    collection_name="fitness_goals",
    embedding_function=embeddings,
    persist_directory="./data/chroma/goals"
)

workout_store = Chroma(
    collection_name="workout_history",
    embedding_function=embeddings,
    persist_directory="./data/chroma/workouts"
)


# State Management
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    health_issues: Optional[List[str]]
    goals: Optional[Dict[str, Any]]
    previous_workouts: Optional[List[Dict[str, Any]]]
    recommended_workout: Optional[Dict[str, Any]]


# Tools for the agents
@tool
def check_health_issues(conversation: str, user_id: str = "default_user") -> Dict[str, Any]:
    """
    Analyzes the conversation to identify any health issues, injuries, or pain points.
    """
    # Search for previous health records for this specific user
    results = health_store.similarity_search(
        f"user:{user_id} {conversation}",
        k=3,
        filter={"user_id": user_id} if user_id != "default_user" else None
    )
    context = "\n".join([doc.page_content for doc in results])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a health monitoring assistant. Analyze the conversation to identify:
        1. Any mentioned injuries or pain points
        2. Affected body parts
        3. Severity levels
        4. Contraindicated exercises

        If nothing is mentioned, return an empty list for health_issues.
        """),
        ("user", f"Conversation: {conversation}\n\nContext from previous records: {context}")
    ])

    response = llm.invoke(prompt.format_messages(conversation=conversation, context=context))
    content = response.content

    # Process and store the health information with user_id metadata
    if content:
        health_store.add_texts(
            texts=[conversation],
            metadatas=[{
                "date": datetime.now().isoformat(),
                "analysis": content,
                "user_id": user_id
            }]
        )

    return {
        "health_issues": content
    }


@tool
def track_fitness_goals(conversation: str, user_id: str = "default_user") -> Dict[str, Any]:
    """
    Tracks and retrieves the user's fitness goals from the conversation.
    """
    # Search for previous goal information for this specific user
    results = goals_store.similarity_search(
        f"user:{user_id} {conversation}",
        k=3,
        filter={"user_id": user_id} if user_id != "default_user" else None
    )
    context = "\n".join([doc.page_content for doc in results])

    # Rest of function remains the same...
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a goal tracking assistant. Analyze the conversation to identify:
        1. Short-term fitness goals
        2. Long-term fitness goals
        3. Target metrics (weight, measurements, performance)
        4. Timeline expectations
        5. Any preferences mentioned

        Update existing goals if new information is provided.
        """),
        ("user", f"Conversation: {conversation}\n\nContext from previous goals: {context}")
    ])

    response = llm.invoke(prompt.format_messages(conversation=conversation, context=context))
    content = response.content

    # Store the updated goals with user_id metadata
    goals_store.add_texts(
        texts=[conversation],
        metadatas=[{
            "date": datetime.now().isoformat(),
            "goals": content,
            "user_id": user_id
        }]
    )

    return {
        "goals": content
    }


@tool
def retrieve_workout_history(user_id: str = "default_user") -> Dict[str, Any]:
    """
    Retrieves the user's previous workout history.
    """
    # Search for previous workouts for this specific user
    query = f"user:{user_id} recent workouts"
    results = workout_store.similarity_search(
        query,
        k=5,
        filter={"user_id": user_id} if user_id != "default_user" else None
    )

    workouts = []
    for doc in results:
        if hasattr(doc, "metadata") and "workout" in doc.metadata:
            workouts.append(doc.metadata["workout"])
        else:
            workouts.append({"content": doc.page_content, "date": "unknown"})

    return {
        "previous_workouts": workouts
    }


@tool
def recommend_workout(
        health_issues: str,
        goals: str,
        previous_workouts: str,
        user_id: str = "default_user",
        user_details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Recommends a workout based on goals, health issues, and previous workouts.
    """
    # Include user details in the prompt if available
    user_info = ""
    if user_details:
        user_info = f"""
        User Information:
        - Age: {user_details.get('age', 'Not specified')}
        - Gender: {user_details.get('gender', 'Not specified')}
        - Height: {user_details.get('height', 'Not specified')} cm
        - Weight: {user_details.get('weight', 'Not specified')} kg
        - Fitness Level: {user_details.get('fitness_level', 'Not specified')}
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fitness programming expert. Create a personalized workout plan that:
        1. Aligns with the user's stated goals
        2. Avoids exercises that could aggravate identified health issues/injuries
        3. Provides variety from previous workouts
        4. Includes specific exercises, sets, reps, and intensity guidelines
        5. Provides clear instructions and form cues
        6. Includes warm-up and cool-down recommendations
        7. Considers the user's demographic information and fitness level

        Format the workout in a clear, structured manner.
        """),
        ("user", f"""
        {user_info}

        Health Issues: {health_issues}

        Fitness Goals: {goals}

        Previous Workouts: {previous_workouts}

        Please recommend an appropriate workout plan.
        """)
    ])

    response = llm.invoke(prompt.format_messages(
        health_issues=health_issues,
        goals=goals,
        previous_workouts=previous_workouts
    ))
    content = response.content

    # Store the recommended workout with user_id metadata
    workout_store.add_texts(
        texts=[content],
        metadatas=[{
            "date": datetime.now().isoformat(),
            "workout": content,
            "health_context": health_issues,
            "goals_context": goals,
            "user_id": user_id
        }]
    )

    return {
        "recommended_workout": content
    }

@tool
def explain_workout_science(workout: str) -> str:
    """
    Provides scientific explanation for the recommended workout.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fitness science educator explaining workouts to an Indian gym member. 
            Explain the scientific benefits of the workout, including:
            1. The physiological adaptations it promotes
            2. Why it's appropriate given the user's goals
            3. How it safely works around any health issues
            4. Evidence-based benefits of the exercise selection

            Make your explanation:
            - Use examples relevant to Indian lifestyles and diet patterns
            - Reference traditional Indian exercise forms like yoga when appropriate
            - Consider common health concerns in the Indian population (like diabetes and heart disease)
            - Use easily available equipment or alternatives common in Indian households
            - Incorporate terminology that resonates with Indian audiences
            - Consider cultural contexts around fitness and body image in India

            Keep your explanation informative but accessible for an Indian audience.
            """),
        ("user", f"Workout: {workout}")
    ])

    response = llm.invoke(prompt.format_messages(workout=workout))
    return response.content


# Fallback handler for when the LLM fails to generate a response
def handle_llm_error(state) -> dict:
    """Handles errors during LLM response generation."""
    error = state.get("error", "Unknown error occurred")

    return {
        "messages": [
            AIMessage(
                content="I'm having trouble processing your request. Could you please rephrase or provide more details?"
            )
        ]
    }


# Error handling for tools
def handle_tool_error(state) -> dict:
    """Handles errors during tool execution."""
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls if "messages" in state and state["messages"] else []

    if tool_calls:
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\nPlease provide more information so I can help you properly.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }
    else:
        return {
            "messages": [
                AIMessage(
                    content="I encountered an issue processing your request. Could you provide more details?"
                )
            ]
        }


def create_tool_node_with_fallback(tools: list) -> dict:
    """Creates a tool node with error handling."""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error"
    )


# Agent nodes with fallback
class OrchestratorAgent:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a specialized Indian fitness coach assistant with expertise in workout planning.
                Your responses must be precise, concise, and actionable. Limit responses to 3-4 sentences maximum.

                Prioritize:
                - For injuries: Recommend immediate modifications and safety precautions
                - For goals: Provide specific, measurable targets and timeframes
                - For workouts: Suggest specific exercises with clear sets/reps/intensity

                Avoid generic advice. Focus on practical guidance that can be implemented immediately.
                Use technical fitness terminology appropriately but ensure it remains accessible.
                """
            ),
            ("placeholder", "{messages}"),
        ])

    def __call__(self, state: State):
        messages = state["messages"]

        try:
            # Process with the orchestrator prompt without tools
            response = self.llm.invoke(self.prompt.format_messages(messages=messages))
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as e:
            # If all else fails, return a generic error message
            return {
                "messages": [
                    AIMessage(
                        content="I'm having trouble processing your request. Let's try a different approach."
                    )
                ]
            }


# State processing functions
def process_health_check(state: State):
    """Process health check results and update state"""
    try:
        last_message = state["messages"][-1]
        if hasattr(last_message, "content") and last_message.content:
            # Update health issues in state
            state["health_issues"] = last_message.content
        return state
    except Exception:
        # Return state unchanged if processing fails
        return state


def process_goals(state: State):
    """Process goals and update state"""
    try:
        last_message = state["messages"][-1]
        if hasattr(last_message, "content") and last_message.content:
            # Update goals in state
            state["goals"] = last_message.content
        return state
    except Exception:
        # Return state unchanged if processing fails
        return state


def process_workout_history(state: State):
    """Process workout history and update state"""
    try:
        last_message = state["messages"][-1]
        if hasattr(last_message, "content") and last_message.content:
            # Update workout history in state
            state["previous_workouts"] = last_message.content
        return state
    except Exception:
        # Return state unchanged if processing fails
        return state


def prepare_workout_recommendation(state: State):
    """Prepare inputs for workout recommendation"""
    try:
        # Prepare the inputs for the recommendation
        health_issues = state.get("health_issues", "No health issues mentioned")
        goals = state.get("goals", "No specific goals mentioned")
        previous_workouts = state.get("previous_workouts", "No previous workout history")
        user_id = state.get("user_id", "default_user")
        user_details = state.get("user_details", {})

        # Create a new message with the recommendation request
        messages = state["messages"]
        messages.append(
            HumanMessage(
                content="Please recommend a workout based on my information",
                additional_kwargs={
                    "health_issues": health_issues,
                    "goals": goals,
                    "previous_workouts": previous_workouts,
                    "user_id": user_id,
                    "user_details": user_details
                }
            )
        )

        return {"messages": messages}
    except Exception:
        # Return state unchanged if processing fails
        return state


def process_workout_recommendation(state: State):
    """Process workout recommendation and update state"""
    try:
        last_message = state["messages"][-1]
        if hasattr(last_message, "content") and last_message.content:
            # Update recommended workout in state
            state["recommended_workout"] = last_message.content
        return state
    except Exception:
        # Return state unchanged if processing fails
        return state


def custom_router(state):
    """Routes messages based on their content."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # If no message or message doesn't have content
    if not last_message or not hasattr(last_message, "content"):
        return "orchestrator"  # Default route

    content = last_message.content.lower()

    # Check for exit/end commands
    if "goodbye" in content or "thank you" in content or "thanks" in content:
        return "end"  # Route to END

    # Simple keyword-based routing
    if "pain" in content or "injury" in content or "hurt" in content:
        return "check_health_issues"
    elif "goal" in content or "target" in content or "aim" in content or "fat loss" in content:
        return "track_fitness_goals"
    elif "workout" in content or "exercise" in content or "train" in content:
        return "workout_history"
    else:
        return "orchestrator"  # Default route


# Build the graph
# In main.py, update the build_fitness_agent_graph function

def build_fitness_agent_graph():
    # Initialize graph
    builder = StateGraph(State)
    orchestrator = OrchestratorAgent(llm)

    # Add nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("health_check", create_tool_node_with_fallback([check_health_issues]))
    builder.add_node("goal_tracker", create_tool_node_with_fallback([track_fitness_goals]))
    builder.add_node("workout_history", create_tool_node_with_fallback([retrieve_workout_history]))
    builder.add_node("workout_recommender", create_tool_node_with_fallback([recommend_workout]))
    builder.add_node("workout_explainer", create_tool_node_with_fallback([explain_workout_science]))

    # Add processing nodes
    builder.add_node("process_health", RunnableLambda(process_health_check))
    builder.add_node("process_goals", RunnableLambda(process_goals))
    builder.add_node("process_history", RunnableLambda(process_workout_history))
    builder.add_node("prepare_recommendation", RunnableLambda(prepare_workout_recommendation))
    builder.add_node("process_recommendation", RunnableLambda(process_workout_recommendation))

    # Add edges with explicit END condition
    builder.add_edge(START, "orchestrator")

    # CRITICAL FIX: Add a way to reach END from orchestrator
    def should_end(state):
        """Determine if we should end the conversation."""
        # Check if we've reached a conclusion or answer
        messages = state.get("messages", [])
        if len(messages) > 10:  # If conversation is getting long
            return True

        # Otherwise continue routing
        return False

    # First check if we should end
    builder.add_conditional_edges(
        "orchestrator",
        should_end,
        {
            True: END,
            False: "route_request"
        }
    )

    # Add a routing node
    builder.add_node("route_request", RunnableLambda(lambda x: x))

    # Then route to appropriate handler if not ending
    builder.add_conditional_edges(
        "route_request",
        custom_router,
        {
            "check_health_issues": "health_check",
            "track_fitness_goals": "goal_tracker",
            "workout_history": "workout_history",
            "orchestrator": "orchestrator",  # Default route
            "end": END  # Add explicit end path
        }
    )

    # Connect tools to processors
    builder.add_edge("health_check", "process_health")
    builder.add_edge("goal_tracker", "process_goals")
    builder.add_edge("workout_history", "process_history")

    # Connect processors back but allow END
    builder.add_edge("process_health", END)  # Allow ending after processing health
    builder.add_edge("process_goals", END)   # Allow ending after processing goals

    builder.add_edge("process_history", "prepare_recommendation")
    builder.add_edge("prepare_recommendation", "workout_recommender")
    builder.add_edge("workout_recommender", "process_recommendation")
    builder.add_edge("process_recommendation", "workout_explainer")
    builder.add_edge("workout_explainer", END)  # End after explanation

    # Compile graph with memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Main execution
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./data/chroma/health", exist_ok=True)
    os.makedirs("./data/chroma/goals", exist_ok=True)
    os.makedirs("./data/chroma/workouts", exist_ok=True)

    # Build the agent graph
    fitness_agent = build_fitness_agent_graph()

    # Start the conversation
    config = {
        "configurable": {
            "thread_id": "fitness-session-1",
        }
    }

    state = {
        "messages": [
            SystemMessage(
                content="I'm your AI fitness coach. I can help create personalized workouts, track your goals, and adapt exercises to any health concerns you have. How can I help you today?")
        ],
        "health_issues": None,
        "goals": None,
        "previous_workouts": None,
        "recommended_workout": None
    }

    print("🏋️‍♀️ AI Fitness Coach is ready. Type your message below (or type 'exit' to quit):\n")

    # Run the conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Keep up the good work!")
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        try:
            # Process through agent graph
            response = fitness_agent.invoke(state, config)
            state = response  # Update state

            # Output assistant response
            for msg in state["messages"]:
                if isinstance(msg, AIMessage) and msg == state["messages"][-1]:
                    print(f"Coach: {msg.content}")
        except Exception as e:
            print(f"Coach: I'm having some technical difficulties. Let's try again with a different question.")
            print(f"Error: {str(e)}")

        print("-" * 50)
