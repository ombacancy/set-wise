import streamlit as st

# Import the API connector
from api_connector import APIConnector

# Initialize API connector
api = APIConnector()

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
        color: #c2c2c2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #c2c2c2;
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
    h1, h2, h3 {
        color: #c2c2c2 !important;
    }
    p {
        color: #c2c2c2 !important;
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

# Title and description
st.markdown("<h1 class='main-header'>AI Fitness Coach</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your personalized workout assistant</p>", unsafe_allow_html=True)

# User authentication
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.user_details = {}
    st.session_state.messages = []
    st.session_state.profile = {}

# User login/registration form
if st.session_state.user_id is None:
    st.markdown("<h2>Welcome! Let's set up your profile</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Create new profile")

        with st.form(key="user_form", clear_on_submit=True):
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=16, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
            fitness_level = st.select_slider(
                "Fitness Level",
                options=["Beginner", "Intermediate", "Advanced"]
            )
        
            submit_button = st.form_submit_button("Create Profile")
        
            if submit_button and name:
                with st.spinner("Creating profile..."):
                    try:
                        user_data = {
                            "name": name,
                            "age": age,
                            "gender": gender,
                            "height": height,
                            "weight": weight,
                            "fitness_level": fitness_level
                        }
        
                        user = api.register_user(user_data)
                        if "user_id" in user:
                            st.session_state.user_id = user["user_id"]
                            st.session_state.user_name = user["name"]
                            st.session_state.user_details = user
                            st.success("Profile created successfully!")
                            st.rerun()
                        else:
                            st.error(f"Invalid response from API: Missing user_id")
                    except Exception as e:
                        st.error(f"{str(e)}")
                        st.info("Please check if the API server is running correctly")

    with col2:
        st.subheader("Log in with existing ID")
        with st.form(key="login_form"):
            user_id = st.text_input("Enter your user ID:", key="login_user_id")
            login_submit = st.form_submit_button("Log In")

            if login_submit and user_id:
                try:
                    # Login user
                    response = api.login_user(user_id)

                    # Store user info in session state
                    st.session_state.user_id = response["user_id"]
                    st.session_state.user_name = response["name"]
                    st.session_state.user_details = response

                    st.success(f"Welcome back, {response['name']}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error logging in: {str(e)}")
else:
    # Show user is logged in
    st.sidebar.success(f"Logged in as: {st.session_state.user_name}")
    if st.sidebar.button("Log Out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Get user profile from API if not already loaded
    if not st.session_state.profile:
        try:
            st.session_state.profile = api.get_user_profile(st.session_state.user_id)

            # Get chat history
            chat_history = api.get_chat_history(st.session_state.user_id)
            st.session_state.messages = chat_history
        except Exception as e:
            st.error(f"Error loading profile: {str(e)}")

    # Sidebar with user profile
    with st.sidebar:
        st.header("Your Fitness Profile")

        # Display user details
        with st.expander("Personal Information", expanded=True):
            user_details = st.session_state.profile.get("personal_info", {})
            st.write(f"**Name:** {user_details.get('name', 'Not provided')}")
            st.write(f"**Age:** {user_details.get('age', 'Not provided')}")
            st.write(f"**Gender:** {user_details.get('gender', 'Not provided')}")
            st.write(f"**Height:** {user_details.get('height', 'Not provided')} cm")
            st.write(f"**Weight:** {user_details.get('weight', 'Not provided')} kg")
            st.write(f"**Fitness Level:** {user_details.get('fitness_level', 'Not provided')}")

        with st.expander("Health Information", expanded=False):
            health_issues = st.session_state.profile.get("health_issues")
            if health_issues:
                st.write(health_issues)
            else:
                st.write("No health issues recorded. Mention any injuries or concerns in chat.")

        with st.expander("Fitness Goals", expanded=False):
            goals = st.session_state.profile.get("goals")
            if goals:
                st.write(goals)
            else:
                st.write("No fitness goals recorded. Share your goals in chat.")

        with st.expander("Workout History", expanded=False):
            previous_workouts = st.session_state.profile.get("previous_workouts")
            if previous_workouts:
                st.write(previous_workouts)
            else:
                st.write("No workout history recorded yet.")

        if st.button("Clear Conversation"):
            try:
                api.clear_chat_history(st.session_state.user_id)
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing chat: {str(e)}")

        st.markdown("---")
        st.markdown("### Quick Prompts")

        try:
            # Get quick prompts from API
            quick_prompts = api.get_quick_prompts()

            for prompt in quick_prompts:
                if st.button(prompt["title"]):
                    st.session_state.quick_prompt = prompt["content"]
                    st.rerun()
        except Exception as e:
            # Use default prompts if API call fails
            default_prompts = {
                "Set fitness goals": "I want to focus on fat loss and muscle toning.",
                "Report injury": "My right shoulder has been hurting during overhead exercises.",
                "Request workout": "Can you suggest a home workout with minimal equipment?",
                "Ask about nutrition": "What should I eat to support muscle growth?"
            }

            for prompt_title, prompt_text in default_prompts.items():
                if st.button(prompt_title):
                    st.session_state.quick_prompt = prompt_text
                    st.rerun()

    # Main chat area
    st.header("Chat with Your AI Fitness Coach")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""<div class='user-message'>
                    <strong style="color:#000000;">You:</strong>
                    <span style="color:#000000;">{msg["content"]}</span>
                </div>""", unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                st.markdown(f"""<div class='bot-message'>
                    <strong style="color:#000000;">Coach:</strong>
                    <span style="color:#000000;">{msg["content"]}</span>
                </div>""", unsafe_allow_html=True)

    # User input
    user_input = st.chat_input("Ask your fitness coach...")

    # Handle quick prompts
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.quick_prompt
        del st.session_state.quick_prompt

    if user_input:
        # Show user message immediately
        st.markdown(f"""<div class='user-message'>
            <strong style="color:#000000;">You:</strong>
            <span style="color:#000000;">{user_input}</span>
        </div>""", unsafe_allow_html=True)

        # Show spinner while processing
        with st.spinner("Coach is thinking..."):
            try:
                # Send message to API
                response = api.chat_with_coach(st.session_state.user_id, user_input)

                # Add messages to session state
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response["content"]})

                # Display response
                st.markdown(f"""<div class='bot-message'>
                    <strong style="color:#000000;">Coach:</strong>
                    <span style="color:#000000;">{response["content"]}</span>
                </div>""", unsafe_allow_html=True)

                # Refresh profile data after chat
                st.session_state.profile = api.get_user_profile(st.session_state.user_id)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                fallback_msg = "I'll help you with that request. What specific details can you share?"
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})

                st.markdown(f"""<div class='bot-message'>
                    <strong style="color:#000000;">Coach:</strong>
                    <span style="color:#000000;">{fallback_msg}</span>
                </div>""", unsafe_allow_html=True)

        # Force a rerun to update the UI
        st.rerun()

# Footer
api_status = "API connected ✅"
try:
    api.check_api_status()
except Exception as e:
    api_status = f"API connection failed ❌: {str(e)}"
    st.error(f"Cannot connect to the API server at {api.base_url}. Please make sure the FastAPI server is running.")

# Show API status in the footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #333333;">
    Powered by LangGraph and FastAPI | Built with Streamlit | {api_status}
</div>
""", unsafe_allow_html=True)
