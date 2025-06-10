from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from utils.models import UserState
from config import GROQ_API_KEY
import json

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b-32768",
    temperature=0.3
)

with open("prompts/parse_intent.jinja") as f:
    PROMPT_TEMPLATE = PromptTemplate.from_template(f.read())

def parse_intent_node(state: UserState) -> UserState:
    prompt = PROMPT_TEMPLATE.format(user_input=state.user_input)
    response = llm.predict(prompt)
    parsed = json.loads(response)
    state.intent = parsed.get("intent", "unknown")
    state.entities = parsed.get("entities", {})
    return state
