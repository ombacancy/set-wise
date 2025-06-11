from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class WorkoutLog(BaseModel):
    date: datetime = Field(default_factory=datetime.utcnow)
    exercise: str
    reps: Optional[int] = None
    sets: Optional[int] = None
    weight: Optional[float] = None
    notes: Optional[str] = None


class UserStatus(BaseModel):
    soreness: Optional[List[str]] = []
    energy: Optional[int] = None  # scale 1-10
    mood: Optional[str] = None


class UserGoal(BaseModel):
    target: str  # e.g., 'build muscle', 'lose fat', 'improve stamina'
    focus_areas: Optional[List[str]] = []  # e.g., ['arms', 'legs']


class IntentPayload(BaseModel):
    intent: Optional[str] = None  # <-- Make it optional
    user_id: str
    raw_input: str
    workout: Optional[WorkoutLog] = None
    status: Optional[UserStatus] = None
    goal: Optional[UserGoal] = None
