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

workout_log_store = Chroma(
    collection_name="workout_logs",
    embedding_function=embeddings,
    persist_directory="./data/chroma/workout_logs"
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


@tool
def log_user_workout(workout_description: str, user_id: str = "default_user") -> Dict[str, Any]:
    """
    Logs a workout that the user reports completing.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a workout tracking assistant. Parse the user's completed workout into a structured format:
        1. Extract exercises performed
        2. Sets, reps, and weights where mentioned
        3. Duration of cardio activities
        4. Overall workout duration if mentioned
        5. Any notes about difficulty or performance

        Format as a clean, structured summary.
        """),
        ("user", f"Workout completed: {workout_description}")
    ])

    response = llm.invoke(prompt.format_messages(workout_description=workout_description))
    parsed_workout = response.content

    # Store the workout log with user_id metadata
    workout_log_store.add_texts(
        texts=[workout_description],
        metadatas=[{
            "date": datetime.now().isoformat(),
            "parsed_workout": parsed_workout,
            "user_id": user_id,
            "type": "user_completed"
        }]
    )

    return {
        "logged_workout": parsed_workout
    }


@tool
def retrieve_workout_logs(user_id: str = "default_user", days: int = 30) -> Dict[str, Any]:
    """
    Retrieves the user's logged workouts from the past specified days.
    """
    # Search for workout logs for this specific user
    query = f"user:{user_id} recent completed workouts"
    results = workout_log_store.similarity_search(
        query,
        k=10,
        filter={"user_id": user_id, "type": "user_completed"} if user_id != "default_user" else {
            "type": "user_completed"}
    )

    logs = []
    for doc in results:
        if hasattr(doc, "metadata") and "parsed_workout" in doc.metadata:
            logs.append({
                "date": doc.metadata.get("date", "unknown"),
                "workout": doc.metadata["parsed_workout"]
            })
        else:
            logs.append({"content": doc.page_content, "date": "unknown"})

    return {
        "workout_logs": logs
    }


@tool
def summarize_workout_logs(user_id: str = "default_user", timeframe: str = "week") -> Dict[str, Any]:
    """
    Generates a summary of the user's workout logs for the specified timeframe.

    Args:
        user_id: The user's unique identifier.
        timeframe: The timeframe to summarize ("week", "month", or "all").
    """
    # Search for workout logs for this specific user
    query = f"user:{user_id} recent completed workouts"
    results = workout_log_store.similarity_search(
        query,
        k=20,  # Get more results for comprehensive summary
        filter={"user_id": user_id, "type": "user_completed"} if user_id != "default_user" else {"type": "user_completed"}
    )

    # Get the logs with dates
    logs = []
    for doc in results:
        if hasattr(doc, "metadata") and "parsed_workout" in doc.metadata:
            logs.append({
                "date": doc.metadata.get("date", "unknown"),
                "workout": doc.metadata["parsed_workout"],
                "raw_description": doc.page_content
            })

    # Format logs as context for the LLM
    logs_context = "\n\n".join([f"Date: {log['date']}\nWorkout: {log['workout']}" for log in logs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a workout analysis assistant. Generate a summary of the user's workouts for the past {timeframe}.
        Include:
        1. Total workouts completed
        2. Body parts/muscle groups trained
        3. Exercise patterns and preferences
        4. Progress indicators where visible
        5. Suggested areas of improvement
        6. Visual representation of workout frequency (as text-based chart)
        
        Format as a clean, structured summary with sections and bullet points.
        """),
        ("user", f"Here are my recent workouts:\n\n{logs_context}")
    ])

    response = llm.invoke(prompt)
    summary = response.content

    return {
        "workout_summary": summary,
        "timeframe": timeframe,
        "workout_count": len(logs)
    }


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


def sanitize_llm_response(content: str) -> str:
    """
    Cleans LLM responses to remove hallucinations and corrupted output.
    """
    # Check for common hallucination patterns
    hallucination_patterns = [
        r'_both.*',
        r'\(Size.*\)',
        r'[A-Z]{3,}\.visit[A-Za-z]+',
        r'.*BuilderFactory.*',
        r'.*Injected.*',
        r'-----.*',
        r'/slider.*'
    ]

    # Clean the content by removing matches
    cleaned_content = content
    for pattern in hallucination_patterns:
        # Find where the hallucination begins
        match = re.search(pattern, cleaned_content)
        if match:
            # Keep only content before the hallucination
            cleaned_content = cleaned_content[:match.start()]

    # Remove trailing commas and whitespace
    cleaned_content = cleaned_content.rstrip(',. \n\t')

    return cleaned_content


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
            # Process with the orchestrator prompt
            response = self.llm.invoke(self.prompt.format_messages(messages=messages))
            content = response.content

            # Clean the response
            cleaned_content = sanitize_llm_response(content)

            # If too much content was removed, request a new response
            if len(cleaned_content) < len(content) * 0.5:
                # Try once more with a stability directive
                retry_response = self.llm.invoke(
                    self.prompt.format_messages(messages=[
                        *messages,
                        SystemMessage(content="Provide a clean, coherent response without technical artifacts.")
                    ])
                )
                cleaned_content = sanitize_llm_response(retry_response.content)

            return {"messages": [AIMessage(content=cleaned_content)]}
        except Exception as e:
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
    elif "i did" in content or "completed" in content or "finished" in content or "today's workout" in content:
        return "log_workout"
    elif "workout" in content or "exercise" in content or "train" in content:
        return "workout_history"
    elif "summary" in content or "progress" in content or "past workouts" in content or "history" in content:
        return "summarize_workouts"
    else:
        return "orchestrator"  # Default route


# Build the graph
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
    builder.add_node("summarize_workouts", create_tool_node_with_fallback([summarize_workout_logs]))
    builder.add_node("log_workout", create_tool_node_with_fallback([log_user_workout]))
    builder.add_node("retrieve_logs", create_tool_node_with_fallback([retrieve_workout_logs]))

    # Add processing nodes
    builder.add_node("process_health", RunnableLambda(process_health_check))
    builder.add_node("process_goals", RunnableLambda(process_goals))
    builder.add_node("process_history", RunnableLambda(process_workout_history))
    builder.add_node("prepare_recommendation", RunnableLambda(prepare_workout_recommendation))
    builder.add_node("process_recommendation", RunnableLambda(process_workout_recommendation))
    builder.add_node("process_workout_log", RunnableLambda(lambda state: {
        **state,
        "logged_workout": state["messages"][-1].content if state["messages"] and hasattr(state["messages"][-1],
                                                                                         "content") else None
    }))

    # Add edges with explicit END condition
    builder.add_edge(START, "orchestrator")

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
            "log_workout": "log_workout",
            "orchestrator": "orchestrator",
            "end": END
        }
    )

    # Connect tools to processors
    builder.add_edge("health_check", "process_health")
    builder.add_edge("goal_tracker", "process_goals")
    builder.add_edge("workout_history", "process_history")
    builder.add_edge("log_workout", "process_workout_log")

    # Connect processors back but allow END
    builder.add_edge("process_health", END)  # Allow ending after processing health
    builder.add_edge("process_goals", END)   # Allow ending after processing goals
    builder.add_edge("process_workout_log", "retrieve_logs")  # After logging, show history
    builder.add_edge("retrieve_logs", END)  # End after showing logs

    # Recommendation flow
    builder.add_edge("process_history", "prepare_recommendation")
    builder.add_edge("prepare_recommendation", "workout_recommender")
    builder.add_edge("workout_recommender", "process_recommendation")
    builder.add_edge("process_recommendation", "workout_explainer")
    builder.add_edge("workout_explainer", END)  # End after explanation

    # Add special connection for personal profile retrieval
    def should_get_user_details(state):
        """Determine if we need to get user details from profile."""
        return "user_details" not in state or not state["user_details"]

    # Add a node to retrieve user details if not present
    builder.add_node("get_user_details", RunnableLambda(lambda state: {
        **state,
        "user_details": state.get("user_details", {}) or {"note": "User details should be injected here"}
    }))

    # Ensure user details are available before recommendation
    builder.add_conditional_edges(
        "prepare_recommendation",
        should_get_user_details,
        {
            True: "get_user_details",
            False: "workout_recommender"
        }
    )

    builder.add_edge("get_user_details", "workout_recommender")

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
