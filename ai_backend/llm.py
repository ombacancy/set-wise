from config import settings
from openai import OpenAI

client = OpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


def chat_completion(messages, model="llama-3.3-70b-versatile", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def simple_prompt(prompt: str, system_prompt: str = "", model="llama-3.3-70b-versatile"):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    return chat_completion(messages, model=model)
