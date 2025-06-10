from pydantic import BaseModel
from typing import List, Optional

class UserState(BaseModel):
    user_id: str
    user_input: str
    intent: Optional[str] = None
    entities: dict = {}
    response: Optional[str] = None
