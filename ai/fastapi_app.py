# ai/fastapi_app.py

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Import from main.py
from main import (
    build_fitness_agent_graph,
    llm,
    health_store,
    goals_store,
    workout_store
)

# Create necessary directories
os.makedirs("./data/chroma/health", exist_ok=True)
os.makedirs("./data/chroma/goals", exist_ok=True)
os.makedirs("./data/chroma/workouts", exist_ok=True)
os.makedirs("./data/users", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="AI Fitness Coach API",
    description="RESTful API for the AI Fitness Coach application",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (should be replaced with Redis or similar in production)
sessions = {}


# Pydantic models
class UserCreate(BaseModel):
    name: str
    age: int = Field(..., ge=16, le=100)
    gender: str
    height: float = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=30, le=250)
    fitness_level: str


class UserLogin(BaseModel):
    user_id: str


class UserResponse(BaseModel):
    user_id: str
    name: str
    age: int
    gender: str
    height: float
    weight: float
    fitness_level: str
    created_at: str


class ChatMessage(BaseModel):
    content: str


class ChatRequest(BaseModel):
    user_id: str
    message: ChatMessage


class MessageResponse(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    messages: List[MessageResponse]


class ProfileInfo(BaseModel):
    personal_info: Dict[str, Any]
    health_issues: Optional[str] = None
    goals: Optional[str] = None
    previous_workouts: Optional[str] = None


# Helper functions
# ai/fastapi_app.py - Fix get_user_details function

def get_user_details(user_id: str) -> dict:
    """Retrieve user details from file storage"""
    if not os.path.exists(f"./data/users/{user_id}.txt"):
        raise HTTPException(status_code=404, detail="User not found")

    user_details = {}
    with open(f"./data/users/{user_id}.txt", "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                user_details[key.strip()] = value.strip()

    # Ensure user_id is correctly mapped
    if "id" in user_details and "user_id" not in user_details:
        user_details["user_id"] = user_details["id"]
    elif "user_id" not in user_details:
        user_details["user_id"] = user_id

    return user_details


def initialize_session(user_id: str, user_details: dict):
    """Initialize or get a session for a user"""
    if user_id not in sessions:
        messages = [
            SystemMessage(
                content=f"I'm your AI fitness coach. I'm here to help you, {user_details.get('name')}. I can create personalized workouts, track your goals, and adapt exercises to any health concerns you have. How can I help you today?")
        ]

        state = {
            "messages": messages,
            "health_issues": None,
            "goals": None,
            "previous_workouts": None,
            "recommended_workout": None,
            "user_id": user_id,
            "user_details": user_details
        }

        config = RunnableConfig(
            configurable={
                "thread_id": f"fitness-session-{user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "user_id": user_id
            },
            recursion_limit=100,
            callbacks=None
        )

        agent = build_fitness_agent_graph()

        sessions[user_id] = {
            "agent": agent,
            "config": config,
            "state": state,
            "history": []
        }

    return sessions[user_id]


# API endpoints
@app.post("/api/users/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Create a new user profile"""
    user_id = str(uuid.uuid4())

    user_details = {
        "user_id": user_id,  # Changed from "id" to "user_id" to match the UserResponse model
        "name": user_data.name,
        "age": user_data.age,
        "gender": user_data.gender,
        "height": user_data.height,
        "weight": user_data.weight,
        "fitness_level": user_data.fitness_level,
        "created_at": datetime.now().isoformat()
    }

    # Save user details to file - we'll keep "id" in the file for backward compatibility
    with open(f"./data/users/{user_id}.txt", "w") as f:
        f.write(f"id: {user_id}\n")  # Keep id for backward compatibility
        f.write(f"user_id: {user_id}\n")  # Add user_id explicitly
        for key, value in user_details.items():
            if key not in ["user_id"]:  # Skip user_id as we already wrote it
                f.write(f"{key}: {value}\n")

    return UserResponse(**user_details)


@app.post("/api/users/login", response_model=UserResponse)
async def login_user(login_data: UserLogin):
    """Log in an existing user"""
    try:
        user_details = get_user_details(login_data.user_id)
        return UserResponse(**user_details)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/profile", response_model=ProfileInfo)
async def get_user_profile(user_id: str):
    """Get user profile information"""
    try:
        user_details = get_user_details(user_id)

        # Initialize session to get state
        session = initialize_session(user_id, user_details)
        state = session["state"]

        profile_info = {
            "personal_info": user_details,
            "health_issues": state.get("health_issues"),
            "goals": state.get("goals"),
            "previous_workouts": state.get("previous_workouts")
        }

        return profile_info
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/{user_id}", response_model=MessageResponse)
async def chat_with_coach(user_id: str, chat_request: ChatMessage):
    """Chat with the AI fitness coach"""
    try:
        # Get user details
        user_details = get_user_details(user_id)

        # Initialize or get the session
        session = initialize_session(user_id, user_details)

        # Add user message to state
        user_message = HumanMessage(content=chat_request.content)
        session["state"]["messages"].append(user_message)
        session["history"].append(user_message)

        try:
            # Process through agent graph
            response = session["agent"].invoke(session["state"], session["config"])
            session["state"] = response  # Update state

            # Find the latest AI message
            new_ai_messages = []
            for msg in session["state"]["messages"]:
                if isinstance(msg, AIMessage) and msg not in session["history"]:
                    new_ai_messages.append(msg)

            # If we got no AI messages, create a fallback response
            if not new_ai_messages:
                fallback_msg = AIMessage(content="I understand your request. Let me help you with that.")
                session["history"].append(fallback_msg)
                session["state"]["messages"].append(fallback_msg)
                return MessageResponse(role="assistant", content=fallback_msg.content)
            else:
                # Add all new AI messages to history
                for msg in new_ai_messages:
                    session["history"].append(msg)

                # Return the latest AI message
                return MessageResponse(role="assistant", content=new_ai_messages[-1].content)

        except Exception as e:
            error_msg = AIMessage(content=f"I'll help you with that request. What specific details can you share?")
            session["history"].append(error_msg)
            session["state"]["messages"].append(error_msg)
            return MessageResponse(role="assistant", content=error_msg.content)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/{user_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(user_id: str):
    """Get chat history for a user"""
    try:
        # Get user details
        user_details = get_user_details(user_id)

        # Initialize or get the session
        session = initialize_session(user_id, user_details)

        # Convert history to response model
        messages = []
        for msg in session["history"]:
            if isinstance(msg, HumanMessage):
                messages.append(MessageResponse(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(MessageResponse(role="assistant", content=msg.content))

        return ChatHistoryResponse(messages=messages)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/{user_id}/clear")
async def clear_chat_history(user_id: str):
    """Clear chat history for a user"""
    try:
        # Get user details
        user_details = get_user_details(user_id)

        # Reset session
        if user_id in sessions:
            del sessions[user_id]

        # Initialize new session
        initialize_session(user_id, user_details)

        return JSONResponse(content={"status": "success", "message": "Chat history cleared"})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quick-prompts")
async def get_quick_prompts():
    """Get a list of quick prompts for the chat"""
    quick_prompts = [
        {"title": "Set fitness goals", "content": "I want to focus on fat loss and muscle toning."},
        {"title": "Report injury", "content": "My right shoulder has been hurting during overhead exercises."},
        {"title": "Request workout", "content": "Can you suggest a home workout with minimal equipment?"},
        {"title": "Ask about nutrition", "content": "What should I eat to support muscle growth?"}
    ]
    return quick_prompts


# Health check endpoint
@app.get("/")
async def root():
    return {"status": "API is running", "version": "1.0.0"}


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
