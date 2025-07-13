import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("MODEL_NAME"),
    temperature=float(os.getenv("TEMPERATURE", 0.7))
)
