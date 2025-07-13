# ai/streamlit_app.py

import os
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime

# Import from main.py
from main import (
    build_fitness_agent_graph,
    llm,
    health_store,
    goals_store,
    workout_store
)

# Page configuration
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="🏋️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style and branding with improved visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3951c6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #e0e5ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        color: #000000 !important;
        border: 1px solid #c0c0c0;
    }
    .bot-message {
        background-color: #d0f0fd;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3951c6;
        color: #000000 !important;
        border: 1px solid #c0c0c0;
    }
    .stChatMessage {
        background-color: transparent !important;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #c0c0c0;
    }
    h1, h2, h3, p {
        color: #333333 !important;
    }
    .stMarkdown {
        color: #333333 !important;
    }
    .user-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Create necessary directories
os.makedirs("./data/chroma/health", exist_ok=True)
os.makedirs("./data/chroma/goals", exist_ok=True)
os.makedirs("./data/chroma/workouts", exist_ok=True)
os.makedirs("./data/users", exist_ok=True)

# Title and description
st.markdown("<h1 class='main-header'>🏋️‍♀️ AI Fitness Coach</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your personalized workout assistant</p>", unsafe_allow_html=True)

# User authentication
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.user_details = {}

# User login/registration form
if st.session_state.user_id is None:
    st.markdown("<h2>Welcome! Let's set up your profile</h2>", unsafe_allow_html=True)

    with st.form(key="user_form", clear_on_submit=True):
        st.markdown('<div class="user-form">', unsafe_allow_html=True)
        user_option = st.radio("Choose an option:", ["Create new profile", "Log in with existing ID"])

        if user_option == "Create new profile":
            user_name = st.text_input("Your name:")
            age = st.number_input("Age:", min_value=16, max_value=100, value=30)
            gender = st.selectbox("Gender:", ["Male", "Female", "Non-binary", "Prefer not to say"])
            height = st.number_input("Height (cm):", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg):", min_value=30, max_value=250, value=70)
            fitness_level = st.select_slider(
                "Fitness Level:",
                options=["Beginner", "Intermediate", "Advanced"]
            )

            submit = st.form_submit_button("Create Profile")

            if submit and user_name:
                # Create new user with UUID
                user_id = str(uuid.uuid4())
                st.session_state.user_id = user_id
                st.session_state.user_name = user_name
                st.session_state.user_details = {
                    "id": user_id,
                    "name": user_name,
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "fitness_level": fitness_level,
                    "created_at": datetime.now().isoformat()
                }

                # Save user details to file
                with open(f"./data/users/{user_id}.txt", "w") as f:
                    for key, value in st.session_state.user_details.items():
                        f.write(f"{key}: {value}\n")

                st.success(f"Profile created successfully! Your ID: {user_id}")
                st.info("Please save your ID for future logins")
                st.rerun()

        else:  # Log in with existing ID
            user_id = st.text_input("Enter your user ID:")
            submit = st.form_submit_button("Log In")

            if submit and user_id:
                # Check if user exists
                if os.path.exists(f"./data/users/{user_id}.txt"):
                    # Load user details
                    user_details = {}
                    with open(f"./data/users/{user_id}.txt", "r") as f:
                        for line in f:
                            if ":" in line:
                                key, value = line.strip().split(":", 1)
                                user_details[key.strip()] = value.strip()

                    st.session_state.user_id = user_id
                    st.session_state.user_name = user_details.get("name", "User")
                    st.session_state.user_details = user_details
                    st.success(f"Welcome back, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("User ID not found. Please check and try again or create a new profile.")

        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show user is logged in
    st.sidebar.success(f"Logged in as: {st.session_state.user_name}")
    st.sidebar.button("Log Out", on_click=lambda: st.session_state.clear())

    # Initialize session state for chat and graph
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"I'm your AI fitness coach. I'm here to help you, {st.session_state.user_name}. I can create personalized workouts, track your goals, and adapt exercises to any health concerns you have. How can I help you today?")
        ]
        st.session_state.history = []
        st.session_state.state = {
            "messages": st.session_state.messages,
            "health_issues": None,
            "goals": None,
            "previous_workouts": None,
            "recommended_workout": None,
            "user_id": st.session_state.user_id,  # Add user_id to state
            "user_details": st.session_state.user_details  # Add user details to state
        }
        st.session_state.agent = build_fitness_agent_graph()
        st.session_state.config = RunnableConfig(
            configurable={
                "thread_id": f"fitness-session-{st.session_state.user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "user_id": st.session_state.user_id  # Add user_id to config
            },
            recursion_limit=100,
            callbacks=None
        )

    # Sidebar with user profile
    with st.sidebar:
        st.header("Your Fitness Profile")

        # Display user details
        with st.expander("Personal Information", expanded=True):
            user_details = st.session_state.user_details
            st.write(f"**Name:** {user_details.get('name', 'Not provided')}")
            st.write(f"**Age:** {user_details.get('age', 'Not provided')}")
            st.write(f"**Gender:** {user_details.get('gender', 'Not provided')}")
            st.write(f"**Height:** {user_details.get('height', 'Not provided')} cm")
            st.write(f"**Weight:** {user_details.get('weight', 'Not provided')} kg")
            st.write(f"**Fitness Level:** {user_details.get('fitness_level', 'Not provided')}")

        with st.expander("Health Information", expanded=False):
            if st.session_state.state["health_issues"]:
                st.write(st.session_state.state["health_issues"])
            else:
                st.write("No health issues recorded yet. Mention any injuries or pain points in the chat.")

        with st.expander("Fitness Goals", expanded=False):
            if st.session_state.state["goals"]:
                st.write(st.session_state.state["goals"])
            else:
                st.write("No goals recorded yet. Share your fitness goals in the chat.")

        with st.expander("Workout History", expanded=False):
            if st.session_state.state["previous_workouts"]:
                st.write(st.session_state.state["previous_workouts"])
            else:
                st.write("No workout history yet.")

        if st.button("Clear Conversation"):
            st.session_state.messages = [
                SystemMessage(content=f"I'm your AI fitness coach. I'm here to help you, {st.session_state.user_name}. I can create personalized workouts, track your goals, and adapt exercises to any health concerns you have. How can I help you today?")
            ]
            st.session_state.history = []
            st.session_state.state = {
                "messages": st.session_state.messages,
                "health_issues": None,
                "goals": None,
                "previous_workouts": None,
                "recommended_workout": None,
                "user_id": st.session_state.user_id,
                "user_details": st.session_state.user_details
            }
            st.rerun()

        st.markdown("---")
        st.markdown("### Quick Prompts")
        quick_prompts = {
            "Set fitness goals": "I want to focus on fat loss and muscle toning.",
            "Report injury": "My right shoulder has been hurting during overhead exercises.",
            "Request workout": "Can you suggest a home workout with minimal equipment?",
            "Ask about nutrition": "What should I eat to support muscle growth?"
        }

        for prompt_title, prompt_text in quick_prompts.items():
            if st.button(prompt_title):
                st.session_state.quick_prompt = prompt_text
                st.rerun()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.history):
            if isinstance(msg, HumanMessage):
                st.markdown(f"""<div class='user-message'>
                    <strong style="color:#000000;">You:</strong>
                    <span style="color:#000000;">{msg.content}</span>
                </div>""", unsafe_allow_html=True)
            elif isinstance(msg, AIMessage):
                st.markdown(f"""<div class='bot-message'>
                    <strong style="color:#000000;">Coach:</strong>
                    <span style="color:#000000;">{msg.content}</span>
                </div>""", unsafe_allow_html=True)

    # User input
    user_input = st.chat_input("Ask your fitness coach...")

    # Handle quick prompts
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.quick_prompt
        del st.session_state.quick_prompt

    if user_input:
        # Add user message to state
        user_message = HumanMessage(content=user_input)
        st.session_state.messages.append(user_message)
        st.session_state.history.append(user_message)
        st.session_state.state["messages"] = st.session_state.messages

        # Show spinner while processing
        with st.spinner("Coach is thinking..."):
            try:
                # Process through agent graph with increased recursion limit
                response = st.session_state.agent.invoke(st.session_state.state, st.session_state.config)
                st.session_state.state = response  # Update state

                # Find the latest AI message
                new_ai_messages = []
                for msg in st.session_state.state["messages"]:
                    if isinstance(msg, AIMessage) and msg not in st.session_state.history:
                        new_ai_messages.append(msg)

                # If we got no AI messages, create a fallback response
                if not new_ai_messages:
                    fallback_msg = AIMessage(content="I understand your request. Let me help you with that.")
                    st.session_state.history.append(fallback_msg)
                    st.session_state.state["messages"].append(fallback_msg)
                else:
                    # Add all new AI messages to history
                    for msg in new_ai_messages:
                        st.session_state.history.append(msg)


                # Update the sidebar info if data is available
                if "health_issues" in st.session_state.state and st.session_state.state["health_issues"]:
                    st.sidebar.success("Health information updated!")

                if "goals" in st.session_state.state and st.session_state.state["goals"]:
                    st.sidebar.success("Fitness goals updated!")

                if "recommended_workout" in st.session_state.state and st.session_state.state["recommended_workout"]:
                    st.sidebar.success("New workout added to your history!")

            except Exception as e:
                error_msg = AIMessage(content=f"I'll help you with that request. What specific details can you share?")
                st.session_state.history.append(error_msg)
                st.session_state.state["messages"].append(error_msg)
                st.error(f"Error: {str(e)}")

        # Rerun to update the UI
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #333333;">
    Powered by LangGraph and ChromaDB | Built with Streamlit
</div>
""", unsafe_allow_html=True)